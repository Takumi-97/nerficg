#!/usr/bin/python
#! -*- encoding: utf-8 -*-

import os
import subprocess
import sys
from pathlib import Path

# --- パス設定 ---
OPENMVG_SFM_BIN = "/home/bandailab/openMVG_build/Linux-x86_64-RELEASE"
CAMERA_SENSOR_WIDTH_DIRECTORY = "/home/bandailab/openMVG/src/openMVG/exif/sensor_width_database"
DATASET_ROOT = "/home/bandailab/nerficg/dataset/OmniBlender"

camera_file_params = os.path.join(CAMERA_SENSOR_WIDTH_DIRECTORY, "sensor_width_camera_database.txt")


def run_sfm(scene: str) -> bool:
    scene_dir = Path(DATASET_ROOT) / scene
    input_dir = str(scene_dir / "images")
    out_dir = scene_dir / "openMVG" / "reconstruction"
    ply_path = out_dir / "colorized.ply"

    if ply_path.exists():
        print(f"[{scene}] colorized.ply already exists, skipping.")
        return True

    matches_dir = str(out_dir / "matches")
    reconstruction_dir = str(out_dir / "sequential")
    os.makedirs(matches_dir, exist_ok=True)
    os.makedirs(reconstruction_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Scene: {scene}")
    print(f"  Input: {input_dir}")
    print(f"{'='*60}")

    steps = [
        ("1. Intrinsics analysis (Spherical)", [
            os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfMInit_ImageListing"),
            "-i", input_dir, "-o", matches_dir,
            "-d", camera_file_params,
            "-c", "7",  # Spherical camera model
            "-f", "1"
        ]),
        ("2. Compute features", [
            os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeFeatures"),
            "-i", matches_dir + "/sfm_data.json", "-o", matches_dir,
            "-m", "SIFT", "-p", "ULTRA", "-f", "1"
        ]),
        ("3. Compute matches", [
            os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeMatches"),
            "-i", matches_dir + "/sfm_data.json",
            "-o", matches_dir + "/matches.putative.bin",
            "-f", "1", "-n", "AUTO"
        ]),
        ("4. Filter matches", [
            os.path.join(OPENMVG_SFM_BIN, "openMVG_main_GeometricFilter"),
            "-i", matches_dir + "/sfm_data.json",
            "-m", matches_dir + "/matches.putative.bin",
            "-g", "f", "-o", matches_dir + "/matches.f.bin"
        ]),
        ("5. Incremental reconstruction", [
            os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfM"),
            "--sfm_engine", "INCREMENTAL",
            "--input_file", matches_dir + "/sfm_data.json",
            "--match_dir", matches_dir,
            "--output_dir", reconstruction_dir
        ]),
        ("6. Colorize structure", [
            os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeSfM_DataColor"),
            "-i", reconstruction_dir + "/sfm_data.bin",
            "-o", str(ply_path)
        ]),
    ]

    for label, cmd in steps:
        print(f"  {label}")
        ret = subprocess.run(cmd).returncode
        if ret != 0:
            print(f"  FAILED at step: {label} (exit {ret})")
            return False

    if ply_path.exists():
        print(f"  Done → {ply_path}")
        return True
    else:
        print(f"  colorized.ply not found after reconstruction.")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run OpenMVG SfM for OmniBlender scenes")
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="Scene names to process (default: all missing)")
    args = parser.parse_args()

    dataset_root = Path(DATASET_ROOT)
    if args.scenes:
        scenes = args.scenes
    else:
        # 全シーンのうちcolorized.plyがないものだけ対象
        scenes = [
            d.name for d in sorted(dataset_root.iterdir())
            if d.is_dir()
            and (d / "transform.json").exists()
            and not (d / "openMVG" / "reconstruction" / "colorized.ply").exists()
        ]

    print(f"Scenes to process: {scenes}")
    failed = []
    for scene in scenes:
        ok = run_sfm(scene)
        if not ok:
            failed.append(scene)

    print(f"\nDone. Failed: {failed if failed else 'none'}")


if __name__ == "__main__":
    main()
