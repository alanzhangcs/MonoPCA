"""Run the released MonoPCA model on images and write KITTI-format predictions.

The weights are downloaded from https://huggingface.co/zhihao406/MonoPCA unless --checkpoint is given.

Example:
    python tools/inference.py --image data/KITTIDataset/training/image_2/000001.png \
        --calib data/KITTIDataset/training/calib/000001.txt --output outputs/inference

    # a folder of images with calib files of the same name (<id>.png <-> <id>.txt)
    python tools/inference.py --image path/to/image_2 --calib path/to/calib --output outputs/inference

Each output line is: type trunc occ alpha x1 y1 x2 y2 h w l x y z ry score (KITTI format, camera coordinates).
"""
import warnings

warnings.filterwarnings("ignore")

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from lib.helpers.model_helper import build_model
from lib.helpers.decode_helper import extract_dets_from_outputs, decode_detections
from lib.datasets.kitti.kitti_utils import Calibration, get_affine_transform

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def load_model(cfg, checkpoint=None, hf_repo="zhihao406/MonoPCA", device="cuda"):
    model, _ = build_model(cfg["model"])
    if checkpoint is None:
        from huggingface_hub import hf_hub_download
        checkpoint = hf_hub_download(hf_repo, "model.safetensors")
    if checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint)
    else:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state"]
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def preprocess(img, calib, resolution, test_focal):
    """Resample the image to the virtual focal length and the network resolution (as in evaluation)."""
    img_size = np.array(img.size)
    center = img_size / 2

    # resize to fit the network resolution, keeping the aspect ratio
    scale = min(resolution[0] / img_size[0], resolution[1] / img_size[1])
    new_img_size = np.array([int(scale * img_size[0]), int(scale * img_size[1])])

    # crop so that the image is seen at the virtual focal length
    if resolution[1] / img_size[1] > resolution[0] / img_size[0]:
        crop_ratio = calib.P2[0, 0] * resolution[0] / test_focal / img_size[0]
    else:
        crop_ratio = calib.P2[0, 0] * resolution[1] / test_focal / img_size[1]
    trans, trans_inv = get_affine_transform(center, crop_ratio * img_size, 0, new_img_size, inv=1)

    img = img.transform(tuple(new_img_size.tolist()), method=Image.AFFINE,
                        data=tuple(trans_inv.reshape(-1).tolist()), resample=Image.BILINEAR)
    padded = Image.new(img.mode, tuple(resolution), (0, 0, 0))
    padded.paste(img, (0, 0))

    x = (np.array(padded).astype(np.float32) / 255.0 - IMAGE_MEAN) / IMAGE_STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)

    P2 = calib.P2.copy()
    P2[0, 0] = test_focal  # the network sees the image at the virtual focal length
    return x, torch.from_numpy(P2).float().unsqueeze(0), trans_inv


@torch.no_grad()
def predict(model, img, calib, cfg, device="cuda"):
    """Returns a list of [cls_id, alpha, x1, y1, x2, y2, h, w, l, x, y, z, ry, score]."""
    infer_cfg = cfg["inference"]
    resolution = np.array(infer_cfg["resolution"])
    x, P2, trans_inv = preprocess(img, calib, resolution, infer_cfg["test_focal"])

    img_sizes = torch.tensor([resolution], dtype=torch.long, device=device)
    target_focal = torch.tensor([[infer_cfg["test_focal"]]], dtype=torch.float32, device=device)
    outputs = model(x.to(device), P2.to(device), target_focal, img_sizes)

    dets = extract_dets_from_outputs(outputs, K=50, topk=infer_cfg["topk"]).cpu().numpy()
    info = {"img_id": [0], "img_size": resolution[None], "trans_inv": trans_inv[None]}
    cls_mean_size = np.zeros((len(infer_cfg["class_names"]), 3), dtype=np.float32)
    return decode_detections(dets, info, [calib], cls_mean_size, threshold=infer_cfg["threshold"], large_object_id=[])[0]


def write_kitti(preds, class_names, path):
    with open(path, "w") as f:
        for pred in preds:
            f.write("{} 0.0 0".format(class_names[int(pred[0])]))
            for v in pred[1:]:
                f.write(" {:.2f}".format(v))
            f.write("\n")


def list_inputs(image, calib):
    if os.path.isdir(image):
        names = sorted(n for n in os.listdir(image) if n.lower().endswith(IMAGE_EXTS))
        return [(os.path.join(image, n), os.path.join(calib, os.path.splitext(n)[0] + ".txt")) for n in names]
    return [(image, calib)]


def main():
    parser = argparse.ArgumentParser(description="MonoPCA inference")
    parser.add_argument("--image", required=True, help="an image or a folder of images")
    parser.add_argument("--calib", required=True, help="a KITTI calib file (with P2) or a folder of them")
    parser.add_argument("--output", required=True, help="folder for the KITTI-format prediction txt files")
    parser.add_argument("--config", default=os.path.join(ROOT_DIR, "configs", "monopca_inference.yaml"))
    parser.add_argument("--checkpoint", default=None, help="local .safetensors/.pth; default: download from Hugging Face")
    parser.add_argument("--hf_repo", default="zhihao406/MonoPCA")
    parser.add_argument("--threshold", type=float, default=None, help="override the confidence threshold")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    if args.threshold is not None:
        cfg["inference"]["threshold"] = args.threshold
    # paths in the config are relative to the repository root
    cfg["model"]["focal_embedding_path"] = os.path.join(ROOT_DIR, cfg["model"]["focal_embedding_path"])

    model = load_model(cfg, args.checkpoint, args.hf_repo)
    os.makedirs(args.output, exist_ok=True)
    inputs = list_inputs(args.image, args.calib)
    for image_path, calib_path in inputs:
        preds = predict(model, Image.open(image_path).convert("RGB"), Calibration(calib_path), cfg)
        name = os.path.splitext(os.path.basename(image_path))[0]
        write_kitti(preds, cfg["inference"]["class_names"], os.path.join(args.output, name + ".txt"))
    print(f"Wrote predictions for {len(inputs)} image(s) to {args.output}")


if __name__ == "__main__":
    main()
