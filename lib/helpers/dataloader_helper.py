import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from lib.datasets.kitti.kitti_and_nusc_and_waymo_dataset_multi_focal_multi_class import KITTI_NUSC_Waymo_Dataset_Multi_Focal_Multi_Class


# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(cfg, workers=12, is_distributed=False):
    # perpare dataset
    if cfg["type"] == "kitti_nusc_waymo_multi_focal_multi_class":
        train_set = KITTI_NUSC_Waymo_Dataset_Multi_Focal_Multi_Class(split=cfg["train_split"], cfg=cfg, data_augmentation=True)
        test_set = KITTI_NUSC_Waymo_Dataset_Multi_Focal_Multi_Class(split=cfg["test_split"], cfg=cfg, data_augmentation=False)
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg["type"])

    if is_distributed:
        train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True)
        test_sampler = DistributedSampler(test_set, shuffle=False)
        shuffle = False  # Let sampler handle shuffling
    else:
        train_sampler = None
        test_sampler = None
        shuffle = True

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg["batch_size"],
        sampler=train_sampler,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=False,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=cfg["batch_size"],
        sampler=test_sampler,
        num_workers=workers,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, test_loader
