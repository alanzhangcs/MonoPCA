import warnings

warnings.filterwarnings("ignore")

import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


parser = argparse.ArgumentParser(description="MonoPCA for Monocular 3D Object Detection")
parser.add_argument("--config", dest="config", help="settings of detection in yaml format")
parser.add_argument("-e", "--evaluate_only", action="store_true", default=False, help="evaluation only")
parser.add_argument("--seed", default=444, type=int, help="seed")
args = parser.parse_args()


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    # ----------- initialize DDP -----------
    
    local_rank = setup_distributed()
    device = torch.device("cuda", local_rank)

    # ----------- config & seed -----------
    assert os.path.exists(args.config)
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    set_random_seed(seed=args.seed + local_rank)
    
    args.use_amp = cfg.get("use_amp", True)
    

    # ----------- set default dtype to float32 for single precision training -----------
    torch.set_default_dtype(torch.float32)
    if local_rank == 0:
        print(f"Training with single precision (float32)")
        if args.use_amp:
            print(f"Using Automatic Mixed Precision (AMP)")

    # ----------- logging only on main process -----------
    # only the main process generates the timestamp
    if not dist.is_initialized() or dist.get_rank() == 0:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        current_time = None

    # multi-GPU: broadcast current_time to all processes
    if dist.is_initialized():
        current_time = [current_time]
        dist.broadcast_object_list(current_time, src=0)
        current_time = current_time[0]

    if local_rank == 0:
        model_name = cfg["model_name"]
        output_path = os.path.join("./" + cfg["trainer"]["save_path"] + model_name + str(args.seed) + f"_{current_time}")
        os.makedirs(output_path, exist_ok=True)
        model_name = cfg["model_name"] + str(args.seed) + f"_{current_time}"
        log_file = os.path.join(output_path, "train.log.%s" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        logger = create_logger(log_file)
    else:
        model_name = cfg["model_name"] + str(args.seed) + f"_{current_time}"
        logger = None

    # ----------- dataloader with distributed sampler -----------
    train_loader, test_loader = build_dataloader(cfg["dataset"], is_distributed=True)

    # ----------- model and loss -----------
    model, loss = build_model(cfg["model"])
    model.to(device)
    # static_graph=True: the computation graph is identical every iteration.
    # find_unused_parameters is incompatible with static_graph and must not be set together.
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, static_graph=True, gradient_as_bucket_view=True)

    # ----------- optimizer & scheduler -----------
    optimizer = build_optimizer(cfg["optimizer"], model)
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg["lr_scheduler"], optimizer, last_epoch=-1)

    # ----------- evaluation only -----------
    if args.evaluate_only:
        if local_rank == 0:
            logger.info("###################  Evaluation Only  ##################")
        tester = Tester(cfg=cfg["tester"], model=model, dataloader=test_loader, logger=logger, train_cfg=cfg["trainer"], model_name=model_name)
        tester.test()
        cleanup_distributed()
        return

    # ----------- training -----------
    trainer = Trainer(cfg=cfg["trainer"], model=model, optimizer=optimizer, train_loader=train_loader, test_loader=test_loader, lr_scheduler=lr_scheduler, warmup_lr_scheduler=warmup_lr_scheduler, logger=logger, loss=loss, model_name=model_name, use_amp=args.use_amp)
    
    tester = Tester(cfg=cfg["tester"], model=trainer.model, dataloader=test_loader, logger=logger, train_cfg=cfg["trainer"], model_name=model_name)

    if cfg["dataset"]["test_split"] != "test":
        trainer.tester = tester

    if local_rank == 0:
        logger.info("###################  Training  ##################")
        logger.info("Batch Size: %d" % (cfg["dataset"]["batch_size"]))
        logger.info("Learning Rate: %f" % (cfg["optimizer"]["lr"]))
        logger.info("Precision: Single (float32)")
        if args.use_amp:
            logger.info("AMP: Enabled")

    trainer.train()

    if local_rank == 0 and cfg["dataset"]["test_split"] != "test":
        logger.info("###################  Testing  ##################")
        logger.info("Batch Size: %d" % (cfg["dataset"]["batch_size"]))
        logger.info("Split: %s" % (cfg["dataset"]["test_split"]))
    tester.test()
    torch.distributed.barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
