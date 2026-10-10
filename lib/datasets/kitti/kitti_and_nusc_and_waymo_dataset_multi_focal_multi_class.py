import os
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFile, ImageEnhance
import random
from skimage import io
import skimage.transform
import cv2
import torch.nn.functional as F
import torch
import pickle

ImageFile.LOAD_TRUNCATED_IMAGES = True

import tqdm
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(ROOT_DIR)
ROOT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(ROOT_DIR)

from lib.datasets.kitti.pd import PhotometricDistort

from lib.datasets.utils import angle2class
from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian
from lib.datasets.kitti.kitti_utils import get_objects_from_label
from lib.datasets.kitti.kitti_utils import Calibration
from lib.datasets.kitti.kitti_utils import get_affine_transform
from lib.datasets.kitti.kitti_utils import affine_transform
from lib.datasets.kitti.kitti_eval_python.eval_omni3d import get_official_eval_result
from lib.datasets.kitti.kitti_eval_python.eval_omni3d import get_distance_eval_result
import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti
import copy
import lmdb


# from .pd import PhotometricDistort
def find_continuous_indices(data_list, dataset_name):
    arr = np.array(data_list)
    mask = arr == dataset_name
    diff = np.diff(mask.astype(int))

    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]

    # handle the beginning and the end
    if mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask[-1]:
        ends = np.append(ends, len(mask) - 1)

    return list(zip(starts, ends))


class KITTI_NUSC_Waymo_Dataset_Multi_Focal_Multi_Class(data.Dataset):
    def __init__(self, split, cfg, data_augmentation=True):

        # basic configuration
        self.root_dir = cfg.get("root_dir")
        self.split = split
        self.num_classes = 11
        self.max_objs = 50
        self.class_name = ["Car", "Pedestrian", "Trailer", "Bus", "Motorcycle", "Cyclist", "Barrier", "Truck", "Van", "Traffic_cone", "Bicycle"]
        self.cls2id = {"Car": 0, "Pedestrian": 1, "Trailer": 2, "Bus": 3, "Motorcycle": 4, "Cyclist": 5, "Barrier": 6, "Truck": 7, "Van": 8, "Traffic_cone": 9, "Bicycle": 10}
        self.resolution = np.array(cfg.get("resolution", [1280, 384]))  # W * H
        self.threshold = cfg.get("threshold", 65)
        self.dataset_list = cfg.get(
            "dataset_list",
            [
                "KITTI",
            ],
        )
        self.root_dir_list = cfg.get(
            "root_dir_list",
            [
                "data/KITTIDataset",
            ],
        )

        assert len(self.dataset_list) == len(self.root_dir_list)

        assert self.split in [
            "train",
            "val",
            "trainval",
            "train_nusc",
            "trainval_kitti",
            "val_waymo_small",
            "test",
            "train_kitti_nusc_10k",
            "train_waymo",
            "val_waymo",
            "train_kitti_nusc",
            "train_kitti_trainval_nusc_waymo_10k",
            "val_kitti",
            "val_waymo",
            "test_kitti",
            "val_nusc",
            "train_kitti_nusc_10k_waymo_10k",
            "train_kitti_nusc_waymo_10k",
            "train_kitti_nusc_waymo",
            "val_waymo_small",
            "train_waymo_10k",
            "train_kitti_trainval_nusc_10k_waymo_10k",
        ]
        self.test_dataset = cfg.get("test_dataset", "KITTI")

        self.data_dir_dict = {}
        for i in range(len(self.dataset_list)):
            dataset_name = self.dataset_list[i]
            root_dir = self.root_dir_list[i]

            if dataset_name == "KITTI":
                data_dir = os.path.join(root_dir, "testing" if split == "test_kitti" else "training")
            elif dataset_name == "NUSC":
                data_dir = os.path.join(root_dir, "validation" if split.startswith("val") else "training")
            elif dataset_name == "Waymo":
                data_dir = os.path.join(root_dir, "validation" if split.startswith("val") else "training")

            self.data_dir_dict[dataset_name] = data_dir

        self.use_3d_center = cfg.get("use_3d_center", True)
        self.writelist = cfg.get("writelist", ["Car"])
        # anno: use src annotations as GT, proj: use projected 2d bboxes as GT
        self.bbox2d_type = cfg.get("bbox2d_type", "anno")
        assert self.bbox2d_type in ["anno", "proj"]
        self.meanshape = cfg.get("meanshape", False)
        self.class_merging = cfg.get("class_merging", False)
        self.use_dontcare = cfg.get("use_dontcare", False)
        self.maintain_image_ratio = cfg.get("maintain_image_ratio", True)
        self.target_focal = cfg.get("target_focal", None)
        self.target_focal_list = cfg.get("target_focal_list", [748.8391264865, 900, 1100, 1300])
        self.test_focal = cfg.get("test_focal", 748.8391264865)

        if self.class_merging:
            self.writelist.extend(["Van", "Truck"])
        if self.use_dontcare:
            self.writelist.extend(["DontCare"])

        # data split loading
        self.split_path = cfg.get("split_path", "data/KITTIDataset/ImageSets")
        self.split_file = os.path.join(self.split_path, self.split + ".txt")

        self.idx_list = [x.strip().split(" ", 1)[1] for x in open(self.split_file).readlines()]
        self.idx_dataset = [x.strip().split(" ", 1)[0] for x in open(self.split_file).readlines()]
        self.dataset_index = {}
        for dataset_name in self.dataset_list:
            index = find_continuous_indices(self.idx_dataset, dataset_name)
            if len(index) > 0:
                self.dataset_index[dataset_name] = index[0]
        # data augmentation configuration
        # self.data_augmentation = True if split in ["train", "trainval", "train_kitti_nusc", "train_kitti_all_nusc", "train_kitti_nusc_10k"] else False
        self.data_augmentation = data_augmentation
        self.aug_pd = cfg.get("aug_pd", False)
        self.aug_crop = cfg.get("aug_crop", False)
        self.aug_calib = cfg.get("aug_calib", False)
        self.random_mixup3d = cfg.get("random_mixup3d", 0.5)
        self.random_flip = cfg.get("random_flip", 0.5)
        self.random_crop = cfg.get("random_crop", 0.5)
        self.scale = cfg.get("scale", 0.4)
        self.shift = cfg.get("shift", 0.1)
        self.filter_waymo = cfg.get("filter_waymo", False)

        self.depth_scale = cfg.get("depth_scale", "normal")

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.cls_mean_size = np.array(
            [
                [1.76255119, 0.66068622, 0.84422524],
                [1.52563191462, 1.62856739989, 3.88311640418],
                [1.73698127, 0.59706367, 1.76282397],
                [1.76255119, 0.66068622, 0.84422524],
                [1.52563191462, 1.62856739989, 3.88311640418],
                [1.73698127, 0.59706367, 1.76282397],
                [1.76255119, 0.66068622, 0.84422524],
                [1.52563191462, 1.62856739989, 3.88311640418],
                [1.73698127, 0.59706367, 1.76282397],
                [1.73698127, 0.59706367, 1.76282397],
                [1.73698127, 0.59706367, 1.76282397],
            ]
        )
        if not self.meanshape:
            self.cls_mean_size = np.zeros_like(self.cls_mean_size, dtype=np.float32)

        # others
        self.pd = PhotometricDistort()
        self.clip_2d = cfg.get("clip_2d", False)

        # cache configuration
        self.use_cache = cfg.get("use_cache", True)
        self.cache_dir = cfg.get("cache_dir", os.path.join("/dev/shm", "MULTI_DATASET_CACHE1"))

        # initialize cache
        self.image_cache = {}
        self.label_cache = {}
        self.calib_cache = {}

        if self.use_cache:
            self._init_cache()

    def _init_cache(self):
        """Initialize LMDB caches (image/label/calib)."""
        os.makedirs(self.cache_dir, exist_ok=True)

        def open_lmdb(name):
            return lmdb.open(os.path.join(self.cache_dir, f"{self.split}_{name}.lmdb"), map_size=200 * 1024 * 1024 * 1024, subdir=True, readonly=False, lock=False, readahead=False, meminit=False)  # 10 GB

        self.image_lmdb_env = open_lmdb("image")
        self.label_lmdb_env = open_lmdb("label")
        self.calib_lmdb_env = open_lmdb("calib")

    def get_image(self, dataset_name, idx):
        cache_key = f"{dataset_name}_{idx}".encode()

        if self.use_cache:
            with self.image_lmdb_env.begin(write=False) as txn:
                value = txn.get(cache_key)
                if value:
                    np_img = pickle.loads(value)
                    return Image.fromarray(np_img)

        image_dir = self.get_image_dir(dataset_name)
        img_file = os.path.join(image_dir, "%06d.png" % idx)
        assert os.path.exists(img_file), f"[Missing image] dataset={dataset_name}, idx={idx}, path={img_file}"

        if dataset_name == "Waymo":
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_np = img
        else:
            img_pil = Image.open(img_file)
            img_np = np.array(img_pil)

        if self.use_cache:
            with self.image_lmdb_env.begin(write=True) as txn:
                txn.put(cache_key, pickle.dumps(img_np))

        return Image.fromarray(img_np)

    def get_label(self, dataset_name, idx):
        cache_key = f"{dataset_name}_{idx}".encode()

        if self.use_cache:
            with self.label_lmdb_env.begin(write=False) as txn:
                value = txn.get(cache_key)
                if value:
                    return pickle.loads(value)

        label_dir = self.get_label_dir(dataset_name)
        label_file = os.path.join(label_dir, "%06d.txt" % idx)
        assert os.path.exists(label_file)
        objects = get_objects_from_label(label_file)

        if self.use_cache:
            with self.label_lmdb_env.begin(write=True) as txn:
                txn.put(cache_key, pickle.dumps(objects))

        return objects

    def get_calib(self, dataset_name, idx):
        cache_key = f"{dataset_name}_{idx}".encode()

        if self.use_cache:
            with self.calib_lmdb_env.begin(write=False) as txn:
                value = txn.get(cache_key)
                if value:
                    return pickle.loads(value)

        calib_dir = self.get_calib_dir(dataset_name)
        calib_file = os.path.join(calib_dir, "%06d.txt" % idx)
        assert os.path.exists(calib_file)
        calib = Calibration(calib_file)

        if self.use_cache:
            with self.calib_lmdb_env.begin(write=True) as txn:
                txn.put(cache_key, pickle.dumps(calib))

        return calib

    def get_image_dir(self, dataset_name):
        assert dataset_name in self.dataset_list

        if dataset_name == "KITTI":
            return os.path.join(self.data_dir_dict[dataset_name], "image_2")
        elif dataset_name == "NUSC":
            return os.path.join(self.data_dir_dict[dataset_name], "image")
        elif dataset_name == "Waymo":
            return os.path.join(self.data_dir_dict[dataset_name], "image")

    def get_label_dir(self, dataset_name):
        assert dataset_name in self.dataset_list

        if dataset_name == "KITTI":
            return os.path.join(self.data_dir_dict[dataset_name], "label_2")
        elif dataset_name == "NUSC":
            return os.path.join(self.data_dir_dict[dataset_name], "label_2")
        elif dataset_name == "Waymo":
            return os.path.join(self.data_dir_dict[dataset_name], "label")

    def get_calib_dir(self, dataset_name):
        assert dataset_name in self.dataset_list

        if dataset_name == "KITTI":
            return os.path.join(self.data_dir_dict[dataset_name], "calib")
        elif dataset_name == "NUSC":
            return os.path.join(self.data_dir_dict[dataset_name], "calib")
        elif dataset_name == "Waymo":
            return os.path.join(self.data_dir_dict[dataset_name], "calib")

    def eval(self, results_dir, logger):
        logger.info("==> Loading detections and GTs...")
        img_ids = [int(id) for id in self.idx_list]
        dt_annos = kitti.get_label_annos(results_dir)
        gt_annos = kitti.get_label_annos(self.get_label_dir(self.test_dataset), img_ids)

        test_id = {"Car": 0, "Pedestrian": 1, "Trailer": 2, "Bus": 3, "Motorcycle": 4, "Cyclist": 5, "Barrier": 6, "Truck": 7, "Van": 8, "Traffic_cone": 9, "Bicycle": 10}

        logger.info("==> Evaluating (official) ...")
        car_moderate = 0
        for category in self.writelist:
            results_str, results_dict, mAP3d_R40 = get_official_eval_result(gt_annos, dt_annos, test_id[category])
            if category == "Car":
                car_moderate = mAP3d_R40
            logger.info(results_str)
        return car_moderate

    def get_image_with_padding(self, img, target_size, pad_color=(0, 0, 0)):
        original_width, original_height = img.size
        target_width, target_height = target_size
        pad_right = target_width - original_width
        pad_bottom = target_height - original_height
        if pad_right < 0 or pad_bottom < 0:
            raise ValueError("target size must be >= the original image size")
        new_img = Image.new(img.mode, (target_width, target_height), pad_color)
        new_img.paste(img, (0, 0))

        return new_img

    def calculate_new_image_size(self, img_size, resolution):
        if resolution[0] / img_size[0] < resolution[1] / img_size[1]:
            return (
                np.array(
                    [
                        int(resolution[0] / img_size[0] * img_size[0]),
                        int(resolution[0] / img_size[0] * img_size[1]),
                    ]
                ),
                resolution[0] / img_size[0],
            )
        else:
            return (
                np.array(
                    [
                        int(resolution[1] / img_size[1] * img_size[0]),
                        int(resolution[1] / img_size[1] * img_size[1]),
                    ]
                ),
                resolution[1] / img_size[1],
            )

    def __len__(self):
        return self.idx_list.__len__()

    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        dataset_name = self.idx_dataset[item]
        # image loading
        img = self.get_image(dataset_name, index)
        img_size = np.array(img.size)

        if self.split != "test":
            dst_W, dst_H = img_size

        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size, crop_scale = img_size, 1
        random_flip_flag, random_crop_flag = False, False
        random_mix_flag = False
        calib = self.get_calib(dataset_name, index)

        if self.data_augmentation:

            if np.random.random() < self.random_mixup3d:
                random_mix_flag = True

            if self.aug_pd:
                img = np.array(img).astype(np.float32)
                img = self.pd(img).astype(np.uint8)
                img = Image.fromarray(img)

            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if self.aug_crop:
                if np.random.random() < self.random_crop:
                    random_crop_flag = True
                    crop_scale = np.clip(
                        np.random.randn() * self.scale + 1,
                        1 - self.scale,
                        1 + self.scale,
                    )
                    crop_size = img_size * crop_scale
                    center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                    center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        if random_mix_flag == True:
            count_num = 0
            random_mix_flag = False
            while count_num < 50:
                count_num += 1
                dataset_index = self.dataset_index[dataset_name]
                random_index = int(np.random.choice(self.idx_list[dataset_index[0] : dataset_index[1]]))
                if random_index == index:
                    continue
                calib_temp = self.get_calib(dataset_name, random_index)

                if calib_temp.cu == calib.cu and calib_temp.cv == calib.cv and calib_temp.fu == calib.fu and calib_temp.fv == calib.fv:
                    img_temp = self.get_image(dataset_name, random_index)
                    img_size_temp = np.array(img_temp.size)
                    dst_W_temp, dst_H_temp = img_size_temp
                    if dst_W_temp == dst_W and dst_H_temp == dst_H:
                        objects_1 = self.get_label(dataset_name, index)
                        objects_2 = self.get_label(dataset_name, random_index)
                        if len(objects_1) + len(objects_2) < self.max_objs:
                            random_mix_flag = True
                            if random_flip_flag == True:
                                img_temp = img_temp.transpose(Image.FLIP_LEFT_RIGHT)
                            img_blend = Image.blend(img, img_temp, alpha=0.5)
                            img = img_blend
                            break

        # add affine transformation for 2d images.

        new_img_size, scale_ratio = self.calculate_new_image_size(img_size, self.resolution)
        if "train" in self.split:
            random_focal = random.choice(self.target_focal_list)
            self.target_focal = random_focal
        else:
            self.target_focal = self.test_focal

        if self.resolution[1] / img_size[1] > self.resolution[0] / img_size[0]:
            crop_ratio_for_target_focal = calib.P2[0, 0] * self.resolution[0] / self.target_focal / img_size[0]
        else:
            crop_ratio_for_target_focal = calib.P2[0, 0] * self.resolution[1] / self.target_focal / img_size[1]
        new_crop_scale = crop_scale - 1 + crop_ratio_for_target_focal
        trans, trans_inv = get_affine_transform(center, new_crop_scale * img_size, 0, new_img_size, inv=1)

        img = img.transform(
            tuple(new_img_size.tolist()),
            method=Image.AFFINE,
            data=tuple(trans_inv.reshape(-1).tolist()),
            resample=Image.BILINEAR,
        )

        img = self.get_image_with_padding(img, self.resolution)

        padded_h, padded_w = self.resolution[1], self.resolution[0]
        orig_h, orig_w = new_img_size[1], new_img_size[0]
        mask_np = np.ones((padded_h, padded_w), dtype=bool)
        mask_np[:orig_h, :orig_w] = False  # valid (non-padding) region is False

        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W

        info = {"img_id": index, "img_size": self.resolution, "trans_inv": trans_inv, "dataset_name": dataset_name}

        if self.split == "test_kitti":
            calib = self.get_calib(dataset_name, index)
            targets = {"target_focals": np.array([self.target_focal], dtype=np.float32)}
            calib.P2[0, 0] = self.target_focal
            return img, calib.P2, targets, info

        #  ============================   get labels   ==============================
        objects = self.get_label(dataset_name, index)
        calib = self.get_calib(dataset_name, index)

        # data augmentation for labels
        if random_flip_flag:
            if self.aug_calib:
                calib.flip(img_size)
            for object in objects:
                [x1, _, x2, _] = object.box2d
                object.box2d[0], object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                object.alpha = np.pi - object.alpha
                object.ry = np.pi - object.ry
                if self.aug_calib:
                    object.pos[0] *= -1
                if object.alpha > np.pi:
                    object.alpha -= 2 * np.pi  # check range
                if object.alpha < -np.pi:
                    object.alpha += 2 * np.pi
                if object.ry > np.pi:
                    object.ry -= 2 * np.pi
                if object.ry < -np.pi:
                    object.ry += 2 * np.pi

        # labels encoding

        calibs = np.zeros((self.max_objs, 3, 4), dtype=np.float32)
        indices = np.zeros((self.max_objs), dtype=np.int64)
        mask_2d = np.zeros((self.max_objs), dtype=bool)
        labels = np.zeros((self.max_objs), dtype=np.int8)
        depth = np.zeros((self.max_objs, 1), dtype=np.float32)
        heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
        heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
        size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
        size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        boxes = np.zeros((self.max_objs, 4), dtype=np.float32)
        boxes_3d = np.zeros((self.max_objs, 6), dtype=np.float32)

        obj_region = np.zeros((img.shape[1], img.shape[2]), dtype=bool)  # (H, W)

        object_num = len(objects) if len(objects) < self.max_objs else self.max_objs

        for i in range(object_num):
            # filter objects by writelist
            if objects[i].cls_type not in self.writelist:
                continue

            # filter inappropriate samples
            if objects[i].level_str == "UnKnown" or objects[i].pos[-1] < 2:
                continue

            # ignore the samples beyond the threshold [hard encoding]
            if objects[i].pos[-1] > self.threshold:
                continue

            # ignore noise label in waymo
            if self.filter_waymo and dataset_name == "Waymo" and objects[i].l > 5:
                continue

            # process 2d bbox & get 2d center
            bbox_2d = objects[i].box2d.copy()

            # add affine transformation for 2d boxes.
            bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
            bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)

            # process 3d center
            center_2d = np.array(
                [(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2],
                dtype=np.float32,
            )  # W * H

            # create object region
            ymin, ymax = int(max(bbox_2d[1], 0)), int(min(bbox_2d[3], img.shape[1]))
            xmin, xmax = int(max(bbox_2d[0], 0)), int(min(bbox_2d[2], img.shape[2]))
            obj_region[ymin:ymax, xmin:xmax] = 1

            corner_2d = bbox_2d.copy()

            center_3d = objects[i].pos + [
                0,
                -objects[i].h / 2,
                0,
            ]  # real 3D center in 3D space
            center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)

            center_3d, rect_depth = calib.rect_to_img(center_3d)  # project 3D center to image plane
            center_3d = center_3d[0]  # shape adjustment

            if random_flip_flag and not self.aug_calib:  # random flip for center3d
                center_3d[0] = img_size[0] - center_3d[0]
            center_3d = affine_transform(center_3d.reshape(-1), trans)

            # filter 3d center out of img
            proj_inside_img = True

            if center_3d[0] < 0 or center_3d[0] >= self.resolution[0]:
                proj_inside_img = False
            if center_3d[1] < 0 or center_3d[1] >= self.resolution[1]:
                proj_inside_img = False

            if proj_inside_img == False:
                continue

            # class
            cls_id = self.cls2id[objects[i].cls_type]
            labels[i] = cls_id

            # encoding 2d/3d boxes
            w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            size_2d[i] = 1.0 * w, 1.0 * h

            center_2d_norm = center_2d / self.resolution
            size_2d_norm = size_2d[i] / self.resolution

            corner_2d_norm = corner_2d
            corner_2d_norm[0:2] = corner_2d[0:2] / self.resolution
            corner_2d_norm[2:4] = corner_2d[2:4] / self.resolution
            center_3d_norm = center_3d / self.resolution

            l, r = (
                center_3d_norm[0] - corner_2d_norm[0],
                corner_2d_norm[2] - center_3d_norm[0],
            )
            t, b = (
                center_3d_norm[1] - corner_2d_norm[1],
                corner_2d_norm[3] - center_3d_norm[1],
            )

            if l < 0 or r < 0 or t < 0 or b < 0:
                if self.clip_2d:
                    l = np.clip(l, 0, 1)
                    r = np.clip(r, 0, 1)
                    t = np.clip(t, 0, 1)
                    b = np.clip(b, 0, 1)
                else:
                    continue

            boxes[i] = (
                center_2d_norm[0],
                center_2d_norm[1],
                size_2d_norm[0],
                size_2d_norm[1],
            )
            boxes_3d[i] = center_3d_norm[0], center_3d_norm[1], l, r, t, b

            # encoding depth
            if self.depth_scale == "normal":
                depth[i] = objects[i].pos[-1] * crop_scale

            elif self.depth_scale == "inverse":
                depth[i] = objects[i].pos[-1] / crop_scale

            elif self.depth_scale == "none":
                depth[i] = objects[i].pos[-1]

            # encoding heading angle
            heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0] + objects[i].box2d[2]) / 2)
            if heading_angle > np.pi:
                heading_angle -= 2 * np.pi  # check range
            if heading_angle < -np.pi:
                heading_angle += 2 * np.pi
            heading_bin[i], heading_res[i] = angle2class(heading_angle)

            # encoding size_3d
            src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
            mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
            size_3d[i] = src_size_3d[i] - mean_size

            if objects[i].trucation <= 0.5 and objects[i].occlusion <= 2:
                mask_2d[i] = 1

            calibs[i] = calib.P2

        if random_mix_flag == True:
            # if False:
            objects = self.get_label(dataset_name, random_index)
            # data augmentation for labels
            if random_flip_flag:
                for object in objects:
                    [x1, _, x2, _] = object.box2d
                    object.box2d[0], object.box2d[2] = (
                        img_size[0] - x2,
                        img_size[0] - x1,
                    )
                    object.ry = np.pi - object.ry
                    object.pos[0] *= -1
                    if object.ry > np.pi:
                        object.ry -= 2 * np.pi
                    if object.ry < -np.pi:
                        object.ry += 2 * np.pi
            object_num_temp = len(objects) if len(objects) < (self.max_objs - object_num) else (self.max_objs - object_num)
            for i in range(object_num_temp):
                if objects[i].cls_type not in self.writelist:
                    continue

                if objects[i].level_str == "UnKnown" or objects[i].pos[-1] < 2:
                    continue
                    # ignore noise label in waymo
                if self.filter_waymo and dataset_name == "Waymo" and objects[i].l > 5:
                    continue
                # process 2d bbox & get 2d center
                bbox_2d = objects[i].box2d.copy()
                # add affine transformation for 2d boxes.
                bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
                bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)

                # process 3d center
                center_2d = np.array(
                    [(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2],
                    dtype=np.float32,
                )  # W * H

                # create object region
                ymin, ymax = int(max(bbox_2d[1], 0)), int(min(bbox_2d[3], img.shape[1]))
                xmin, xmax = int(max(bbox_2d[0], 0)), int(min(bbox_2d[2], img.shape[2]))
                obj_region[ymin:ymax, xmin:xmax] = 1

                corner_2d = bbox_2d.copy()

                center_3d = objects[i].pos + [
                    0,
                    -objects[i].h / 2,
                    0,
                ]  # real 3D center in 3D space
                center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
                center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
                center_3d = center_3d[0]  # shape adjustment
                if random_flip_flag and not self.aug_calib:  # random flip for center3d
                    center_3d[0] = img_size[0] - center_3d[0]
                center_3d = affine_transform(center_3d.reshape(-1), trans)

                # filter 3d center out of img
                proj_inside_img = True

                if center_3d[0] < 0 or center_3d[0] >= self.resolution[0]:
                    proj_inside_img = False
                if center_3d[1] < 0 or center_3d[1] >= self.resolution[1]:
                    proj_inside_img = False

                if proj_inside_img == False:
                    continue

                # class
                cls_id = self.cls2id[objects[i].cls_type]
                labels[i + object_num] = cls_id

                # encoding 2d/3d boxes
                w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
                size_2d[i + object_num] = 1.0 * w, 1.0 * h

                center_2d_norm = center_2d / self.resolution
                size_2d_norm = size_2d[i + object_num] / self.resolution

                corner_2d_norm = corner_2d
                corner_2d_norm[0:2] = corner_2d[0:2] / self.resolution
                corner_2d_norm[2:4] = corner_2d[2:4] / self.resolution
                center_3d_norm = center_3d / self.resolution

                l, r = (
                    center_3d_norm[0] - corner_2d_norm[0],
                    corner_2d_norm[2] - center_3d_norm[0],
                )
                t, b = (
                    center_3d_norm[1] - corner_2d_norm[1],
                    corner_2d_norm[3] - center_3d_norm[1],
                )

                if l < 0 or r < 0 or t < 0 or b < 0:
                    if self.clip_2d:
                        l = np.clip(l, 0, 1)
                        r = np.clip(r, 0, 1)
                        t = np.clip(t, 0, 1)
                        b = np.clip(b, 0, 1)
                    else:
                        continue

                boxes[i + object_num] = (
                    center_2d_norm[0],
                    center_2d_norm[1],
                    size_2d_norm[0],
                    size_2d_norm[1],
                )
                boxes_3d[i + object_num] = (
                    center_3d_norm[0],
                    center_3d_norm[1],
                    l,
                    r,
                    t,
                    b,
                )

                # encoding depth
                if self.depth_scale == "normal":
                    depth[i + object_num] = objects[i].pos[-1] * crop_scale

                elif self.depth_scale == "inverse":
                    depth[i + object_num] = objects[i].pos[-1] / crop_scale

                elif self.depth_scale == "none":
                    depth[i + object_num] = objects[i].pos[-1]

                # encoding heading angle
                # heading_angle = objects[i].alpha
                heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0] + objects[i].box2d[2]) / 2)
                if heading_angle > np.pi:
                    heading_angle -= 2 * np.pi  # check range
                if heading_angle < -np.pi:
                    heading_angle += 2 * np.pi
                heading_bin[i + object_num], heading_res[i + object_num] = angle2class(heading_angle)

                # offset_3d[i + object_num] = center_3d - center_heatmap
                src_size_3d[i + object_num] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
                mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
                size_3d[i + object_num] = src_size_3d[i + object_num] - mean_size

                if objects[i].trucation <= 0.5 and objects[i].occlusion <= 2:
                    mask_2d[i + object_num] = 1

                calibs[i + object_num] = calib.P2

        # collect return data
        inputs = img
        if self.maintain_image_ratio:
            img_size = self.resolution
            calib.P2[0, 0] = calib.P2[0, 0] * scale_ratio
            if self.target_focal is not None:
                calib.P2[0, 0] = self.target_focal

        targets = {
            "calibs": calibs,
            "indices": indices,
            "img_size": img_size,
            "labels": labels,
            "boxes": boxes,
            "boxes_3d": boxes_3d,
            "depth": depth,
            "size_2d": size_2d,
            "size_3d": size_3d,
            "src_size_3d": src_size_3d,
            "heading_bin": heading_bin,
            "heading_res": heading_res,
            "mask_2d": mask_2d,
            "obj_region": obj_region,
            "image_mask": mask_np,
            "target_focals": np.array([self.target_focal], dtype=np.float32),
        }

        info = {"img_id": index, "img_size": img_size, "dataset_name": dataset_name}
        if self.maintain_image_ratio:
            info["trans_inv"] = trans_inv
            info["img_size"] = self.resolution

        return (
            inputs,
            calib.P2,
            targets,
            info,
        )
