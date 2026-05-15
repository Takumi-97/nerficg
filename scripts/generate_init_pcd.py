#!/usr/bin/env python3
"""Generate a random initial point cloud for scenes missing colorized.ply.

Uses GT camera positions from transform.json to estimate scene scale,
then creates a random point cloud filling a sphere of that scale.
"""

import json
import struct
import numpy as np
from pathlib import Path
import argparse


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    n = len(points)
    header = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        f"property float x\nproperty float y\nproperty float z\n"
        f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"end_header\n"
    ).encode()
    with open(path, "wb") as f:
        f.write(header)
        for p, c in zip(points, colors):
            f.write(struct.pack("<fff", *p))
            f.write(struct.pack("<BBB", *c))


def generate_for_scene(dataset_dir: Path, scene: str, n_points: int = 50_000) -> None:
    scene_dir = dataset_dir / scene
    recon_dir = scene_dir / "openMVG" / "reconstruction"
    ply_path = recon_dir / "colorized.ply"

    if ply_path.exists():
        print(f"  {scene}: already has colorized.ply, skipping")
        return

    recon_dir.mkdir(parents=True, exist_ok=True)

    with open(scene_dir / "transform.json") as f:
        data = json.load(f)

    # camera centers from GT poses (Blender → world)
    centers = []
    for frame in data["frames"]:
        T = np.array(frame["transform_matrix"])
        centers.append(T[:3, 3])
    centers = np.array(centers)

    centroid = centers.mean(axis=0)
    # scene radius: 3× the max distance of any camera from centroid
    cam_radii = np.linalg.norm(centers - centroid, axis=1)
    scene_radius = max(cam_radii.max() * 3.0, 1.0)

    rng = np.random.default_rng(42)
    # uniform random inside sphere via rejection sampling
    pts = []
    while len(pts) < n_points:
        batch = rng.uniform(-1, 1, (n_points * 2, 3))
        mask = np.linalg.norm(batch, axis=1) <= 1.0
        pts.append(batch[mask])
    pts = np.concatenate(pts)[:n_points] * scene_radius + centroid

    colors = rng.integers(128, 200, size=(n_points, 3), dtype=np.uint8)

    write_ply(ply_path, pts.astype(np.float32), colors)
    print(f"  {scene}: centroid={centroid.round(2)}, radius={scene_radius:.2f} → {n_points} pts → {ply_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="dataset/OmniBlender")
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--n_points", type=int, default=50_000)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if args.scenes:
        scenes = args.scenes
    else:
        scenes = [d.name for d in sorted(dataset_dir.iterdir())
                  if d.is_dir() and (d / "transform.json").exists()]

    print(f"Generating init point clouds for {len(scenes)} scene(s):")
    for scene in scenes:
        generate_for_scene(dataset_dir, scene, args.n_points)
    print("Done.")


if __name__ == "__main__":
    main()
