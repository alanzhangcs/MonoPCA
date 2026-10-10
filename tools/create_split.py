"""Merge per-dataset split files into a single multi-dataset split.

Each line of the output is "<DATASET> <sample_id>", e.g. "KITTI 000003".

Example:
    python tools/create_split.py \
        --inputs KITTI:data/KITTIDataset/ImageSets/train.txt \
                 NUSC:data/NUSCDataset/ImageSets/train.txt \
                 Waymo:data/WaymoDataset/ImageSets/train.txt \
        --output splits/train_kitti_nusc_waymo.txt
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Create a merged multi-dataset split file")
    parser.add_argument("--inputs", nargs="+", required=True, help="list of NAME:path/to/split.txt")
    parser.add_argument("--output", required=True, help="output split file")
    args = parser.parse_args()

    merged = []
    for item in args.inputs:
        name, path = item.split(":", 1)
        with open(path) as f:
            merged += [f"{name} {line.strip()}\n" for line in f if line.strip()]

    with open(args.output, "w") as f:
        f.writelines(merged)
    print(f"Wrote {len(merged)} samples to {args.output}")


if __name__ == "__main__":
    main()
