import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched
import math
import warnings


def build_lr_scheduler(cfg, optimizer, last_epoch):
    """
    Build learning rate scheduler with enhanced options
    """
    scheduler_type = cfg.get('type', 'step')
    
    if scheduler_type == 'step':
        return build_step_scheduler(cfg, optimizer, last_epoch)
    elif scheduler_type == 'cos':
        return build_cosine_scheduler(cfg, optimizer, last_epoch)
    elif scheduler_type == 'cosine_annealing':
        return build_cosine_annealing_scheduler(cfg, optimizer, last_epoch)
    elif scheduler_type == 'onecycle':
        return build_onecycle_scheduler(cfg, optimizer, last_epoch)
    elif scheduler_type == 'reduce_on_plateau':
        return build_reduce_on_plateau_scheduler(cfg, optimizer, last_epoch)
    elif scheduler_type == 'exponential':
        return build_exponential_scheduler(cfg, optimizer, last_epoch)
    elif scheduler_type == 'polynomial':
        return build_polynomial_scheduler(cfg, optimizer, last_epoch)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def build_step_scheduler(cfg, optimizer, last_epoch):
    """Build step decay scheduler (original implementation)"""
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg['decay_list']:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg['decay_rate']
        return cur_decay

    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)
    warmup_lr_scheduler = build_warmup_scheduler(cfg, optimizer)
    return lr_scheduler, warmup_lr_scheduler


def build_cosine_scheduler(cfg, optimizer, last_epoch):
    """Build cosine annealing scheduler"""
    max_epochs = cfg.get('max_epochs', 250)
    min_lr = cfg.get('min_lr', 1e-6)
    
    def cosine_lambda(epoch):
        if epoch < cfg.get('warmup_epochs', 0):
            return epoch / cfg.get('warmup_epochs', 1)
        
        # Cosine annealing
        progress = (epoch - cfg.get('warmup_epochs', 0)) / (max_epochs - cfg.get('warmup_epochs', 0))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (1 - min_lr) * cosine_decay
    
    lr_scheduler = lr_sched.LambdaLR(optimizer, cosine_lambda, last_epoch=last_epoch)
    warmup_lr_scheduler = None  # Warmup is handled in the lambda function
    return lr_scheduler, warmup_lr_scheduler


def build_cosine_annealing_scheduler(cfg, optimizer, last_epoch):
    """Build PyTorch's CosineAnnealingLR scheduler with enhanced configuration"""
    # Support both T_max and max_epochs for backward compatibility
    T_max = cfg.get('T_max', cfg.get('max_epochs', 250))
    eta_min = cfg.get('eta_min', cfg.get('min_lr', 1e-6))
    
    lr_scheduler = lr_sched.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch)
    warmup_lr_scheduler = build_warmup_scheduler(cfg, optimizer)
    return lr_scheduler, warmup_lr_scheduler


def build_onecycle_scheduler(cfg, optimizer, last_epoch):
    """Build OneCycleLR scheduler"""
    max_epochs = cfg.get('max_epochs', 250)
    steps_per_epoch = cfg.get('steps_per_epoch', 1000)
    pct_start = cfg.get('pct_start', 0.3)
    anneal_strategy = cfg.get('anneal_strategy', 'cos')
    div_factor = cfg.get('div_factor', 25.0)
    final_div_factor = cfg.get('final_div_factor', 1e4)
    
    lr_scheduler = lr_sched.OneCycleLR(
        optimizer,
        max_lr=cfg.get('max_lr', 0.001),
        epochs=max_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,
        anneal_strategy=anneal_strategy,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        last_epoch=last_epoch
    )
    warmup_lr_scheduler = None  # OneCycleLR handles warmup internally
    return lr_scheduler, warmup_lr_scheduler


def build_reduce_on_plateau_scheduler(cfg, optimizer, last_epoch):
    """Build ReduceLROnPlateau scheduler"""
    mode = cfg.get('mode', 'min')
    factor = cfg.get('factor', 0.1)
    patience = cfg.get('patience', 10)
    verbose = cfg.get('verbose', True)
    threshold = cfg.get('threshold', 1e-4)
    threshold_mode = cfg.get('threshold_mode', 'rel')
    cooldown = cfg.get('cooldown', 0)
    min_lr = cfg.get('min_lr', 0)
    eps = cfg.get('eps', 1e-8)
    
    lr_scheduler = lr_sched.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        verbose=verbose,
        threshold=threshold,
        threshold_mode=threshold_mode,
        cooldown=cooldown,
        min_lr=min_lr,
        eps=eps
    )
    warmup_lr_scheduler = build_warmup_scheduler(cfg, optimizer)
    return lr_scheduler, warmup_lr_scheduler


def build_exponential_scheduler(cfg, optimizer, last_epoch):
    """Build exponential decay scheduler"""
    gamma = cfg.get('gamma', 0.95)
    
    lr_scheduler = lr_sched.ExponentialLR(optimizer, gamma=gamma, last_epoch=last_epoch)
    warmup_lr_scheduler = build_warmup_scheduler(cfg, optimizer)
    return lr_scheduler, warmup_lr_scheduler


def build_polynomial_scheduler(cfg, optimizer, last_epoch):
    """Build polynomial decay scheduler with proper warmup"""
    max_epochs = cfg.get('max_epochs', 250)
    power = cfg.get('power', 1.0)
    warmup_epochs = cfg.get('warmup_epochs', 0)
    init_lr = cfg.get('init_lr', 1e-5)
    
    def polynomial_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup from init_lr to base_lr
            if warmup_epochs > 0:
                return init_lr + (1.0 - init_lr) * (epoch / warmup_epochs)
            else:
                return 1.0
        
        # Polynomial decay
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        return (1 - progress) ** power
    
    lr_scheduler = lr_sched.LambdaLR(optimizer, polynomial_lambda, last_epoch=last_epoch)
    warmup_lr_scheduler = None  # Warmup is handled in the lambda function
    return lr_scheduler, warmup_lr_scheduler


def build_warmup_scheduler(cfg, optimizer):
    """Build warmup scheduler"""
    if not cfg.get('warmup', False):
        return None
    
    warmup_type = cfg.get('warmup_type', 'cosine')
    warmup_epochs = cfg.get('warmup_epochs', 5)
    init_lr = cfg.get('init_lr', 1e-5)
    
    if warmup_type == 'cosine':
        return CosineWarmupLR(optimizer, num_epoch=warmup_epochs, init_lr=init_lr)
    elif warmup_type == 'linear':
        return LinearWarmupLR(optimizer, num_epoch=warmup_epochs, init_lr=init_lr)
    else:
        raise ValueError(f"Unknown warmup type: {warmup_type}")


def build_bnm_scheduler(cfg, model, last_epoch):
    if not cfg['enabled']:
        return None

    def bnm_lmbd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg['decay_list']:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg['decay_rate']
        return max(cfg['momentum']*cur_decay, cfg['clip'])

    bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd, last_epoch=last_epoch)
    return bnm_scheduler


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(type(model).__name__))

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))


class CosineWarmupLR(lr_sched._LRScheduler):
    def __init__(self, optimizer, num_epoch, init_lr=0.0, last_epoch=-1):
        self.num_epoch = num_epoch
        self.init_lr = init_lr
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.init_lr + (base_lr - self.init_lr) *
                (1 - math.cos(math.pi * self.last_epoch / self.num_epoch)) / 2
                for base_lr in self.base_lrs]


class LinearWarmupLR(lr_sched._LRScheduler):
    def __init__(self, optimizer, num_epoch, init_lr=0.0, last_epoch=-1):
        self.num_epoch = num_epoch
        self.init_lr = init_lr
        super(LinearWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.init_lr + (base_lr - self.init_lr) * self.last_epoch / self.num_epoch
                for base_lr in self.base_lrs]


class CosineAnnealingWarmRestartsLR(lr_sched._LRScheduler):
    """Cosine annealing scheduler with warm restarts"""
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        super(CosineAnnealingWarmRestartsLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]

        if self.T_mult == 1:
            T_cur = self.last_epoch % self.T_0
        else:
            T_cur = self.last_epoch
            while T_cur >= self.T_0:
                T_cur -= self.T_0
                self.T_0 *= self.T_mult

        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * T_cur / self.T_0)) / 2
                for base_lr in self.base_lrs]


def get_lr_scheduler_info(scheduler, warmup_scheduler):
    """Get information about the current scheduler configuration"""
    info = {}
    
    if scheduler is not None:
        info['main_scheduler'] = type(scheduler).__name__
        if hasattr(scheduler, 'get_last_lr'):
            info['current_lr'] = scheduler.get_last_lr()
    
    if warmup_scheduler is not None:
        info['warmup_scheduler'] = type(warmup_scheduler).__name__
        if hasattr(warmup_scheduler, 'get_last_lr'):
            info['warmup_lr'] = warmup_scheduler.get_last_lr()
    
    return info
