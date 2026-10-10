import os
import torch
from utils.misc import is_main_process


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_checkpoint_state(model=None, optimizer=None, epoch=None, best_result=None, best_epoch=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {"epoch": epoch, "model_state": model_state, "optimizer_state": optim_state, "best_result": best_result, "best_epoch": best_epoch}


def save_checkpoint(state, filename):
    filename = "{}.pth".format(filename)
    torch.save(state, filename)


def load_checkpoint(model, optimizer, filename, map_location, logger=None):
    if os.path.isfile(filename):
        if is_main_process() and logger is not None:
            logger.info("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
        epoch = checkpoint.get("epoch", -1)
        best_result = checkpoint.get("best_result", 0.0)
        best_epoch = checkpoint.get("best_epoch", 0.0)
        model_state = checkpoint.get("model_state", None)
        if model is not None and model_state is not None:
            if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                model.module.load_state_dict(model_state)
            else:
                model.load_state_dict(model_state)
        optimizer_state = checkpoint.get("optimizer_state", None)
        if optimizer is not None and optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        if is_main_process() and logger is not None:
            logger.info("==> Done")
    else:
        raise FileNotFoundError("Checkpoint file not found: {}".format(filename))

    return epoch, best_result, best_epoch
