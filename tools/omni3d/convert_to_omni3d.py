"""Convert KITTI-format predictions to the Omni3D prediction format.

Each prediction file `<sample_id>.txt` holds one object per line in KITTI format with a trailing score:
    type trunc occ alpha x1 y1 x2 y2 h w l x y z ry score

The output is a JSON list with one entry per image, which `evaluate_omni3d.py` scores against the
Omni3D test annotations (KITTI_test / nuScenes_test).

Example:
    python tools/omni3d/convert_to_omni3d.py --dataset kitti \
        --pred_dir outputs/.../outputs/data \
        --data_root data/KITTIDataset/training \
        --omni3d_json /path/to/omni3d/datasets/Omni3D/KITTI_test.json \
        --output kitti_omni3d_predictions.json

    python tools/omni3d/convert_to_omni3d.py --dataset nuscenes \
        --pred_dir outputs/.../outputs/data \
        --data_root data/NUSCDataset/validation \
        --omni3d_json /path/to/omni3d/datasets/Omni3D/nuScenes_test.json \
        --output nuscenes_omni3d_predictions.json
"""
import argparse
import copy
import json
import math
import os
import re

import numpy as np
from PIL import Image
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Contiguous category ids of the Omni3D-OUT model (see category_meta.json), per source dataset.
# Categories not listed for a dataset are not annotated in it and are dropped.
CONTIGUOUS_IDS = {
    "kitti": {"pedestrian": 0, "person": 0, "car": 1, "cyclist": 2, "van": 3, "truck": 4},
    "nuscenes": {"pedestrian": 0, "car": 1, "truck": 4, "traffic cone": 5, "barrier": 6,
                 "motorcycle": 7, "bicycle": 8, "bus": 9, "trailer": 10},
}

DATASET_CONFIGS = {
    "kitti": {"image_list": "kitti_val.txt", "image_subdir": "image_2"},
    "nuscenes": {"image_list": "nuscenes_val.txt", "image_subdir": "image"},
}


def fix_category_name(category_name):
    if category_name == "person_sitting":
        category_name = "person"
    if category_name == "traffic_cone":
        category_name = "traffic cone"
    return category_name


def read_lines(path):
    with open(path) as f:
        return [line.rstrip("\n") for line in f]


def get_calib_from_file(calib_file):
    """Read the P2 projection matrix from a KITTI calib file."""
    lines = read_lines(calib_file)
    line = lines[0] if len(lines) <= 2 else lines[2]
    P2 = np.array(line.strip().split(" ")[1:], dtype=np.float32)
    return P2.reshape(3, 4)


def fix_intrinsic(K):
    # make sure the focal lengths are positive
    K_fix = copy.deepcopy(K)
    K_fix[0, 0] = np.abs(K[0, 0])
    K_fix[1, 1] = np.abs(K[1, 1])
    return K_fix


def get_extrinsic(K, P2):
    """Rotation and translation from the projection matrix P2 = K [R | t]."""
    P2_big = np.eye(4)
    P2_big[:3] = P2
    K_big = np.eye(4)
    K_big[:3, :3] = K
    extrinsic = np.matmul(np.linalg.inv(K_big), P2_big)
    return extrinsic[:3, :3], extrinsic[:3, 3]


def r_cam(yaw):
    return [[math.cos(yaw), 0.0, math.sin(yaw)], [0.0, 1.0, 0.0], [-math.sin(yaw), 0.0, math.cos(yaw)]]


def center_to_corners(center, yaw, dimensions):
    """8 box corners in the Omni3D vertex order, see
    https://github.com/facebookresearch/omni3d/blob/main/DATA.md#coordinate-system"""
    half_width = dimensions[0] / 2.0
    half_height = dimensions[1] / 2.0
    half_length = dimensions[2] / 2.0
    local_vertices = np.array(
        [
            [-half_length, -half_height, -half_width],
            [half_length, -half_height, -half_width],
            [half_length, half_height, -half_width],
            [-half_length, half_height, -half_width],
            [-half_length, -half_height, half_width],
            [half_length, -half_height, half_width],
            [half_length, half_height, half_width],
            [-half_length, half_height, half_width],
        ]
    )
    global_vertices = np.dot(r_cam(yaw), local_vertices.T).T
    return (global_vertices + center).tolist()


def project_box(K, corners):
    """Project 3D corners (8 x 3) with intrinsics K and return the enclosing 2D box [x1, y1, x2, y2]."""
    K_big = np.eye(4)
    K_big[:3, :3] = K
    points_4d = np.vstack((np.array(corners).T, np.ones((1, 8))))
    coord2d = np.matmul(K_big, points_4d)
    ind = np.where(np.abs(coord2d[2]) > 1e-2)
    coord2d[:2, ind] /= coord2d[2, ind]
    return [np.min(coord2d[0]), np.min(coord2d[1]), np.max(coord2d[0]), np.max(coord2d[1])]


def trunc_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > width:
        x2 = width
    if y2 > height:
        y2 = height
    return [x1, y1, x2, y2]


def get_image_info(sample_id, data_root, image_subdir):
    img_width, img_height = Image.open(os.path.join(data_root, image_subdir, sample_id + ".png")).size
    P2 = get_calib_from_file(os.path.join(data_root, "calib", sample_id + ".txt"))
    K = fix_intrinsic(P2[:, :3])
    rotation, ext_translation = get_extrinsic(K, P2)
    return {"img_width": img_width, "img_height": img_height, "K": K,
            "rotation": rotation, "ext_translation": ext_translation}


def get_image_id_map(dataset, image_list, omni3d_json):
    """Map KITTI-style sample ids to Omni3D image ids."""
    with open(omni3d_json) as f:
        images = json.load(f)["images"]
    path_to_id = {}
    for img_info in images:
        path_to_id.setdefault(img_info["file_path"], img_info["id"])

    if dataset == "kitti":
        sample_to_path = {i: "KITTI_object/training/image_2/" + i + ".png" for i in image_list}
    else:
        # nuScenes samples exported by DEVIANT are renamed to 6-digit ids; map them back to the original images
        with open(os.path.join(DATA_DIR, "seqid2path_nuScenes.json")) as f:
            seqid2path = json.load(f)
        sample_to_path = {i: re.sub(r"^nuscenes(?=/)", "nuScenes", seqid2path[i]) for i in image_list}

    image_id_map = {}
    for sample_id, path in sample_to_path.items():
        if path in path_to_id:
            image_id_map[sample_id] = path_to_id[path]
        else:
            print(f"[WARN] Cannot find image_id for {path}")
    return image_id_map


def convert_file(txt_path, image_id, image_info, contiguous_ids):
    rotation = image_info["rotation"]
    ext_translation = image_info["ext_translation"]
    K = image_info["K"]

    instances = []
    for line in read_lines(txt_path):
        parts = line.strip().split()
        if len(parts) != 16:
            continue
        category = parts[0].lower()
        trunc, occ, alpha, u1, v1, u2, v2, h, w, l, x, y, z, yaw, score = map(float, parts[1:])

        if category == "misc" or trunc == 1 or u2 - u1 <= 0 or v2 - v1 <= 0 or z <= 0:
            continue
        category = fix_category_name(category)
        if category not in contiguous_ids:
            continue

        # KITTI location is the bottom center of the box; Omni3D uses the 3D center
        center = np.matmul(rotation, np.array([x, y, z]) + ext_translation) - np.array([0.0, h / 2.0, 0.0])
        center = center.tolist()
        dims = [w, h, l]
        bbox3D = center_to_corners(center, yaw, dims)
        x1, y1, x2, y2 = trunc_bbox(project_box(K, bbox3D), image_info["img_width"], image_info["img_height"])
        bbox2D = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]  # XYWH

        center_2d_homo = np.dot(K, np.array(center))
        center_2D = [center_2d_homo[0] / center_2d_homo[2], center_2d_homo[1] / center_2d_homo[2]]

        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        pose = [[cos_yaw, 0.0, sin_yaw], [0.0, 1.0, 0.0], [-sin_yaw, 0.0, cos_yaw]]

        instances.append({
            "image_id": image_id,
            "category_id": contiguous_ids[category],
            "bbox": bbox2D,
            "score": score,
            "depth": np.array(bbox3D)[:, 2].mean(),
            "bbox3D": bbox3D,
            "center_cam": center,
            "center_2D": center_2D,
            "dimensions": dims,
            "pose": pose,
        })
    return instances


def main():
    parser = argparse.ArgumentParser(description="Convert KITTI-format predictions to Omni3D format")
    parser.add_argument("--dataset", choices=["kitti", "nuscenes"], required=True)
    parser.add_argument("--pred_dir", required=True, help="folder of KITTI-format prediction txt files")
    parser.add_argument("--data_root", required=True,
                        help="KITTI-format split folder with image_2/ (KITTI) or image/ (nuScenes) and calib/")
    parser.add_argument("--omni3d_json", required=True, help="Omni3D annotation json, e.g. datasets/Omni3D/KITTI_test.json")
    parser.add_argument("--output", required=True, help="output prediction json")
    args = parser.parse_args()

    config = DATASET_CONFIGS[args.dataset]
    image_list = read_lines(os.path.join(DATA_DIR, config["image_list"]))
    image_id_map = get_image_id_map(args.dataset, image_list, args.omni3d_json)
    contiguous_ids = CONTIGUOUS_IDS[args.dataset]

    predictions = []
    for file_name in tqdm(sorted(os.listdir(args.pred_dir))):
        if not file_name.endswith(".txt"):
            continue
        sample_id = file_name[:-len(".txt")]
        if sample_id not in image_id_map:
            print(f"[SKIP] {sample_id} not in the Omni3D test split")
            continue
        image_id = image_id_map[sample_id]
        image_info = get_image_info(sample_id, args.data_root, config["image_subdir"])
        predictions.append({
            "image_id": image_id,
            "K": image_info["K"].tolist(),
            "width": image_info["img_width"],
            "height": image_info["img_height"],
            "instances": convert_file(os.path.join(args.pred_dir, file_name), image_id, image_info, contiguous_ids),
        })

    with open(args.output, "w") as f:
        json.dump(predictions, f)
    print(f"Saved {len(predictions)} images to {args.output}")


if __name__ == "__main__":
    main()
