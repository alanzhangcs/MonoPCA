import math
import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from functools import partial

# import get_vit_lr_decay_rate
try:
    from lib.models.monopca.backbones.eva02.vit import get_vit_lr_decay_rate
except ImportError:
    # fallback implementation if the import fails
    def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
        layer_id = num_layers + 1
        if name.startswith("backbone"):
            if ".pos_embed" in name or ".patch_embed" in name:
                layer_id = 0
            elif ".blocks." in name and ".residual." not in name:
                layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
        return lr_decay_rate ** (num_layers + 1 - layer_id)



def build_optimizer(cfg_optimizer, model):
    weights, biases = [], []
    for name, param in model.named_parameters():
        if "bias" in name:
            biases += [param]
        else:
            weights += [param]

    parameters = [
        {"params": biases, "weight_decay": 0},
        {"params": weights, "weight_decay": cfg_optimizer["weight_decay"]},
    ]

    if cfg_optimizer["type"] == "sgd":
        optimizer = optim.SGD(parameters, lr=cfg_optimizer["lr"], momentum=0.9)
    elif cfg_optimizer["type"] == "adam":
        optimizer = optim.Adam(parameters, lr=cfg_optimizer["lr"])
    elif cfg_optimizer["type"] == "adamw":
        optimizer = AdamW(parameters, lr=cfg_optimizer["lr"])
    elif cfg_optimizer["type"] == "mob_adamw":
        # supports layer-wise learning-rate decay
        use_layer_decay = cfg_optimizer.get("use_layer_decay", False)
        lr_decay_rate = cfg_optimizer.get("lr_decay_rate", 0.8)
        num_layers = cfg_optimizer.get("num_layers", 24)
        backbone_prefixes = tuple(
            cfg_optimizer.get("backbone_prefixes", ["backbone", "mob.vfm_backbone"])
        )
        
        # check required arguments
        if "backbone_lr" not in cfg_optimizer:
            raise ValueError("backbone_lr is required for mob_adamw optimizer")
        
        # get model parameters, with or without a DDP `module` wrapper
        if hasattr(model, 'module'):
            named_params = model.module.named_parameters()
        else:
            named_params = model.named_parameters()
        
        if use_layer_decay:
            # build parameter groups with layer-wise lr decay
            param_groups = []
            layer_params = {}
            other_params = []
            
            # collect and group all parameters in a single pass
            for name, param in named_params:
                if name.startswith(backbone_prefixes):
                    matched_prefix = next(
                        (p for p in backbone_prefixes if name.startswith(p)), None
                    )
                    name_for_decay = (
                        "backbone" + name[len(matched_prefix) :]
                        if matched_prefix is not None
                        else name
                    )
                    # lr decay rate of this layer
                    decay_rate = get_vit_lr_decay_rate(
                        name_for_decay, lr_decay_rate, num_layers
                    )
                    layer_lr = cfg_optimizer["backbone_lr"] * decay_rate
                    # print(f"Layer [{name}] → LR: {layer_lr:.6e}")
                    
                    # use the rounded lr as key to avoid float precision issues
                    lr_rounded = round(layer_lr, 10)  # keep 10 decimal places
                    if lr_rounded not in layer_params:
                        layer_params[lr_rounded] = {
                            "params": [],
                            "lr": layer_lr,
                            "weight_decay": cfg_optimizer["weight_decay"]
                        }
                    layer_params[lr_rounded]["params"].append(param)
                else:
                    other_params.append(param)
                    
            # add backbone parameter groups (non-empty only)
            for lr_rounded, group in layer_params.items():
                if group["params"]:  # non-empty groups only
                    param_groups.append(group)
            
            # add the remaining parameters
            if other_params:  # non-empty groups only
                param_groups.append({
                    "params": other_params,
                    "lr": cfg_optimizer["lr"],
                    "weight_decay": cfg_optimizer["weight_decay"]
                })
            
            if not param_groups:
                raise ValueError("No parameters found for optimization")
                
            optimizer = optim.AdamW(param_groups)
        else:
            # simple two-group split
            backbone_params = []
            other_params = []
            
            # split parameters in a single pass
            for name, param in named_params:
                if name.startswith(backbone_prefixes):
                    backbone_params.append(param)
                else:
                    other_params.append(param)
            
            param_dicts = []
            if backbone_params:
                param_dicts.append({
                    "params": backbone_params, 
                    "lr": cfg_optimizer["backbone_lr"],
                    "weight_decay": cfg_optimizer["weight_decay"]
                })
            if other_params:
                param_dicts.append({
                    "params": other_params, 
                    "lr": cfg_optimizer["lr"],
                    "weight_decay": cfg_optimizer["weight_decay"]
                })
            
            if not param_dicts:
                raise ValueError("No parameters found for optimization")
                
            optimizer = optim.AdamW(param_dicts)

    else:
        raise NotImplementedError("%s optimizer is not supported" % cfg_optimizer["type"])

    return optimizer




class AdamW(Optimizer):
    """Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                amsgrad = group["amsgrad"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                # p.data.addcdiv_(-step_size, exp_avg, denom)
                p.data.add_(
                    -step_size,
                    torch.mul(p.data, group["weight_decay"]).addcdiv_(1, exp_avg, denom),
                )

        return loss
