# -- coding: utf-8 --

"""
Datasets/OmniBlender.py: Provides a dataset class for scenes from EgoNeRF's OmniBlender dataset.
Data available at: https://www.changwoon.info/publications/EgoNeRF.
"""

import torch
import json
import numpy as np
import math
import os
from natsort import natsorted
from collections import namedtuple

import Framework
from Cameras.Equirectangular import EquirectangularCamera
from Cameras.utils import CameraProperties
from Datasets.Base import BaseDataset
from Datasets.utils import CameraCoordinateSystemsTransformations, loadImagesParallel, WorldCoordinateSystemTransformations
from Datasets.Colmap import fetchPly, quaternion_to_R


OpenMVGImage = namedtuple("OpenMVGImage", ["name", "c2w"])


def rotmat2qvec(R):
    [[Rxx, Ryx, Rzx], [Rxy, Ryy, Rzy], [Rxz, Ryz, Rzz]] = R
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


@Framework.Configurable.configure(
    PATH='dataset/OmniBlender/barbershop',
    BACKGROUND_COLOR=[0.0, 0.0, 0.0],
    NEAR_PLANE=0.1,
    FAR_PLANE=1000.0,
    FOCAL_ANGLE=0.0,
    USE_GT_POSES=True,
)
class CustomDataset(BaseDataset):
    """Dataset class for OmniBlender scenes.

    USE_GT_POSES=True (default): loads camera poses from transform.json (Blender ground-truth).
    USE_GT_POSES=False: loads camera poses from openMVG/data_openmvg_{subset}.json (SfM reconstruction).
    """

    def __init__(self, path: str) -> None:
        # Coordinate systems are determined in load() based on USE_GT_POSES.
        super().__init__(
            path,
            EquirectangularCamera(0.1, 1000.0),
            None,
            None,
        )

    def load(self) -> dict[str, list[CameraProperties] | None]:
        """Loads the dataset into a dict containing lists of CameraProperties for training and testing."""
        self.camera.near_plane = self.NEAR_PLANE
        self.camera.far_plane = self.FAR_PLANE
        dataset: dict[str, list[CameraProperties]] = {subset: [] for subset in self.subsets}

        if self.USE_GT_POSES:
            self._load_gt_poses(dataset)
            self.camera_coordinate_system = CameraCoordinateSystemsTransformations.RIGHT_HAND
            self.world_coordinate_system = WorldCoordinateSystemTransformations.XZY
            # GT poses have no associated SfM point cloud → random initialization
            self.point_cloud = None
        else:
            self._load_openmvg_poses(dataset)
            self.camera_coordinate_system = CameraCoordinateSystemsTransformations.LEFT_HAND
            self.world_coordinate_system = WorldCoordinateSystemTransformations.XnZY
            ply_path = self.dataset_path / 'openMVG' / 'reconstruction' / 'colorized.ply'
            if not os.path.exists(ply_path):
                Framework.Logger.logWarning(f'SfM point cloud not found at "{ply_path}", using random initialization.')
                self.point_cloud = None
            else:
                try:
                    self.point_cloud = fetchPly(ply_path)
                except Exception:
                    raise Framework.DatasetError(f'Failed to load SfM point cloud')

        return dataset

    def _load_gt_poses(self, dataset: dict) -> None:
        """Load camera poses from transform.json (Blender ground-truth c2w matrices)."""
        transform_file = self.dataset_path / 'transform.json'
        with open(transform_file) as f:
            tf = json.load(f)
        frames = tf['frames']
        orig_width: int = tf['width']
        orig_height: int = tf['height']
        focal = orig_width / (2 * math.pi * math.cos(self.FOCAL_ANGLE / 180 * math.pi))

        for subset in self.subsets:
            if subset == 'val':
                continue
            split_file = self.dataset_path / f'{subset}.txt'
            with open(split_file) as f:
                indices = [int(line.strip()) for line in f if line.strip()]

            image_filenames = [str(self.dataset_path / 'images' / frames[i]['file_path']) for i in indices]
            rgbs, alphas = loadImagesParallel(image_filenames, self.IMAGE_SCALE_FACTOR, num_threads=-1, desc=subset)

            for frame_idx, rgb, alpha in zip(indices, rgbs, alphas):
                c2w = torch.as_tensor(frames[frame_idx]['transform_matrix'], dtype=torch.float32)
                focal_x = focal * (rgb.shape[2] / orig_width)
                focal_y = focal * (rgb.shape[1] / orig_height)
                dataset[subset].append(CameraProperties(
                    width=rgb.shape[2],
                    height=rgb.shape[1],
                    rgb=rgb,
                    alpha=alpha,
                    c2w=c2w,
                    focal_x=focal_x,
                    focal_y=focal_y,
                ))

    def _load_openmvg_poses(self, dataset: dict) -> None:
        """Load camera poses from openMVG SfM reconstruction."""
        for subset in self.subsets:
            if subset == 'val':
                continue

            openmvg_file = self.dataset_path / 'openMVG' / f'data_openmvg_{subset}.json'
            with open(openmvg_file) as f:
                openmvg_data = json.load(f)
            camera_info = openmvg_data["intrinsics"][0]["value"]
            original_width = camera_info["ptr_wrapper"]["data"]["value0"]["width"]
            original_height = camera_info["ptr_wrapper"]["data"]["value0"]["height"]
            original_focal = original_width / (2 * math.pi * math.cos(self.FOCAL_ANGLE / 180 * math.pi))
            images = []
            for view in openmvg_data["views"]:
                info = view["value"]
                pose_id = info["ptr_wrapper"]["data"]["id_pose"]
                extrinsics = openmvg_data["extrinsics"][pose_id]["value"]
                c2w = torch.eye(4)
                c2w[:3, :3] = torch.from_numpy(quaternion_to_R(rotmat2qvec(extrinsics["rotation"]))).T
                c2w[:3, 3] = torch.tensor(extrinsics["center"], dtype=torch.float32)
                image_name = info["ptr_wrapper"]["data"]["filename"]
                images.append(OpenMVGImage(name=image_name, c2w=c2w))
            images = natsorted(images, key=lambda data: data.name)
            image_filenames = [str(self.dataset_path / 'images' / image.name) for image in images]
            rgbs, alphas = loadImagesParallel(image_filenames, self.IMAGE_SCALE_FACTOR, num_threads=-1, desc=subset)
            for image, rgb, alpha in zip(images, rgbs, alphas):
                focal_x = original_focal * (rgb.shape[2] / original_width)
                focal_y = original_focal * (rgb.shape[1] / original_height)
                dataset[subset].append(CameraProperties(
                    width=rgb.shape[2],
                    height=rgb.shape[1],
                    rgb=rgb,
                    alpha=alpha,
                    c2w=image.c2w,
                    focal_x=focal_x,
                    focal_y=focal_y,
                ))
