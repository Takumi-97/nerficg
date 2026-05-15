#!/usr/bin/env python3
"""Generate data_openmvg_train.json and data_openmvg_test.json from GT poses in transform.json."""

import os
import json
import numpy as np
import argparse
from pathlib import Path

ptr_wrapper_id = 2147483649
polymorphic_id = 1073741824


def prepare_scene(dataset_dir: str, scene: str) -> None:
    scene_dir = Path(dataset_dir) / scene
    openmvg_dir = scene_dir / "openMVG"
    openmvg_dir.mkdir(exist_ok=True)

    with open(scene_dir / "transform.json") as f:
        frames_file = json.load(f)
    frames = frames_file["frames"]
    img_width = frames_file["width"]
    img_height = frames_file["height"]
    img_dir = str(scene_dir / "images")

    train_indices = set(Path(scene_dir / "train.txt").read_text().splitlines())
    test_indices = set(Path(scene_dir / "test.txt").read_text().splitlines())

    train_views, train_extrs = [], []
    test_views, test_extrs = [], []
    train_idx = test_idx = 0

    for frame in frames:
        file_name = frame["file_path"]
        img_idx = os.path.splitext(file_name)[0]

        Twc = np.array(frame["transform_matrix"])
        Twc[1:3, :] *= -1
        Rwc = Twc[:3, :3]
        twc = Twc[:3, 3]
        Rcw = np.linalg.inv(Rwc)
        tcw = -Rcw @ twc
        center = twc.tolist()
        rotation = Rcw.tolist()

        if img_idx in train_indices:
            train_views.append({
                "key": train_idx,
                "value": {
                    "polymorphic_id": polymorphic_id,
                    "ptr_wrapper": {
                        "id": ptr_wrapper_id + train_idx,
                        "data": {
                            "local_path": "",
                            "filename": file_name,
                            "width": img_width,
                            "height": img_height,
                            "id_view": train_idx,
                            "id_intrinsic": 0,
                            "id_pose": train_idx,
                        },
                    },
                },
            })
            train_extrs.append({"key": train_idx, "value": {"rotation": rotation, "center": center}})
            train_idx += 1
        elif img_idx in test_indices:
            test_views.append({
                "key": test_idx,
                "value": {
                    "polymorphic_id": polymorphic_id,
                    "ptr_wrapper": {
                        "id": ptr_wrapper_id + test_idx,
                        "data": {
                            "local_path": "",
                            "filename": file_name,
                            "width": img_width,
                            "height": img_height,
                            "id_view": test_idx,
                            "id_intrinsic": 0,
                            "id_pose": test_idx,
                        },
                    },
                },
            })
            test_extrs.append({"key": test_idx, "value": {"rotation": rotation, "center": center}})
            test_idx += 1

    def make_intrinsic(n_idx):
        return [{
            "key": 0,
            "value": {
                "polymorphic_id": ptr_wrapper_id,
                "polymorphic_name": "spherical",
                "ptr_wrapper": {
                    "id": ptr_wrapper_id + n_idx,
                    "data": {"value0": {"width": img_width, "height": img_height}},
                },
            },
        }]

    def make_content(views, intrs, extrs):
        return {
            "sfm_data_version": "0.3",
            "root_path": img_dir,
            "views": views,
            "intrinsics": intrs,
            "extrinsics": extrs,
            "structure": [],
            "control_points": [],
        }

    train_path = openmvg_dir / "data_openmvg_train.json"
    test_path = openmvg_dir / "data_openmvg_test.json"

    with open(train_path, "w") as f:
        json.dump(make_content(train_views, make_intrinsic(train_idx), train_extrs), f)
    with open(test_path, "w") as f:
        json.dump(make_content(test_views, make_intrinsic(test_idx), test_extrs), f)

    print(f"  {scene}: {train_idx} train, {test_idx} test → {openmvg_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="dataset/OmniBlender")
    parser.add_argument("--scenes", nargs="+", default=None)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if args.scenes:
        scenes = args.scenes
    else:
        scenes = [d.name for d in sorted(dataset_dir.iterdir())
                  if d.is_dir() and (d / "transform.json").exists()
                  and not (d / "openMVG" / "data_openmvg_train.json").exists()]

    print(f"Preparing {len(scenes)} scene(s):")
    for scene in scenes:
        prepare_scene(str(dataset_dir), scene)
    print("Done.")


if __name__ == "__main__":
    main()
