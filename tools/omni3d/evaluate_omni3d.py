"""Evaluate Omni3D-format predictions with the official Omni3D evaluator, without running Cube R-CNN.

Requires a clone of https://github.com/facebookresearch/omni3d with its environment installed and the
Omni3D annotation jsons downloaded to <omni3d_root>/datasets/Omni3D/.

Example:
    python tools/omni3d/evaluate_omni3d.py --omni3d_root /path/to/omni3d \
        --kitti kitti_omni3d_predictions.json \
        --nuscenes nuscenes_omni3d_predictions.json \
        --output_dir outputs/omni3d_eval
"""
import argparse
import json
import os
import sys

import numpy as np

# the official Omni3D evaluator uses np.float, which was removed in numpy 1.24
if not hasattr(np, "float"):
    np.float = float

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Omni3D-format predictions")
    parser.add_argument("--omni3d_root", required=True, help="path to a clone of facebookresearch/omni3d")
    parser.add_argument("--kitti", help="predictions on KITTI_test (from convert_to_omni3d.py)")
    parser.add_argument("--nuscenes", help="predictions on nuScenes_test (from convert_to_omni3d.py)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default=os.path.join(CONFIG_DIR, "omni3d_out.yaml"))
    parser.add_argument("--category_meta", default=os.path.join(CONFIG_DIR, "category_meta.json"))
    args = parser.parse_args()

    predictions = {name: os.path.abspath(path) for name, path in
                   [("KITTI_test", args.kitti), ("nuScenes_test", args.nuscenes)] if path}
    assert predictions, "pass at least one of --kitti / --nuscenes"
    output_dir = os.path.abspath(args.output_dir)
    config = os.path.abspath(args.config)
    category_meta = os.path.abspath(args.category_meta)

    # Omni3D resolves datasets/Omni3D/*.json relative to its root
    omni3d_root = os.path.abspath(args.omni3d_root)
    os.chdir(omni3d_root)
    sys.path.insert(0, omni3d_root)

    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.utils.logger import setup_logger
    from cubercnn.config import get_cfg_defaults
    from cubercnn.data import simple_register
    from cubercnn.evaluation import Omni3DEvaluationHelper
    from cubercnn import data, util

    cfg = get_cfg()
    get_cfg_defaults(cfg)
    cfg.merge_from_file(os.path.join(omni3d_root, "configs", "Base.yaml"))
    cfg.merge_from_file(config)
    cfg.DATASETS.TEST = tuple(predictions)
    cfg.OUTPUT_DIR = output_dir
    cfg.freeze()

    os.makedirs(output_dir, exist_ok=True)
    setup_logger(output=output_dir, name="cubercnn")

    filter_settings = data.get_filter_settings_from_cfg(cfg)
    for dataset_name in predictions:
        simple_register(dataset_name, filter_settings, filter_empty=False)

    # categories of the Omni3D-OUT model the predictions are expressed in
    metadata = util.load_json(category_meta)
    MetadataCatalog.get("omni3d_model").thing_classes = metadata["thing_classes"]
    MetadataCatalog.get("omni3d_model").thing_dataset_id_to_contiguous_id = {
        int(key): val for key, val in metadata["thing_dataset_id_to_contiguous_id"].items()}

    # loading each dataset once registers its annotation json in MetadataCatalog for the evaluator
    for dataset_name in predictions:
        DatasetCatalog.get(dataset_name)

    # evaluation filters, as in Cube R-CNN's do_test
    filter_settings["visibility_thres"] = cfg.TEST.VISIBILITY_THRES
    filter_settings["truncation_thres"] = cfg.TEST.TRUNCATION_THRES
    filter_settings["min_height_thres"] = 0.0625
    filter_settings["max_depth"] = 1e8

    eval_helper = Omni3DEvaluationHelper(
        list(predictions), filter_settings, os.path.join(output_dir, "inference", "iter_final"),
        iter_label="final", only_2d=cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_3D == 0.0)

    for dataset_name, path in predictions.items():
        with open(path) as f:
            results_json = json.load(f)
        eval_helper.add_predictions(dataset_name, results_json)
        eval_helper.save_predictions(dataset_name)
        eval_helper.evaluate(dataset_name)

    eval_helper.summarize_all()


if __name__ == "__main__":
    main()
