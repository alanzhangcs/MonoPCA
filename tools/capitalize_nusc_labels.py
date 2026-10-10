"""Capitalize nuScenes class names exported by DEVIANT (e.g. car -> Car, traffic_cone -> Traffic_cone).

Reads <split>/label and writes <split>/label_2, which is what the MonoPCA dataset loads.

Example:
    python tools/capitalize_nusc_labels.py --root data/NUSCDataset
"""
import argparse
import glob
import os


def main():
    parser = argparse.ArgumentParser(description="Capitalize nuScenes class names in KITTI-format labels")
    parser.add_argument("--root", default="data/NUSCDataset", help="nuScenes root containing training/ and validation/")
    args = parser.parse_args()

    for split in ["training", "validation"]:
        input_folder = os.path.join(args.root, split, "label")
        output_folder = os.path.join(args.root, split, "label_2")
        os.makedirs(output_folder, exist_ok=True)

        txt_files = glob.glob(os.path.join(input_folder, "*.txt"))
        for file_path in txt_files:
            with open(file_path) as infile, open(os.path.join(output_folder, os.path.basename(file_path)), "w") as outfile:
                for line in infile:
                    outfile.write(line[0].upper() + line[1:])
        print(f"{split}: wrote {len(txt_files)} files to {output_folder}")


if __name__ == "__main__":
    main()
