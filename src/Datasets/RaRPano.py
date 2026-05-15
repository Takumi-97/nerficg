# -- coding: utf-8 --

"""
Datasets/RaRPano.py: Provides a dataset class for panorama scenes from the Rounding and Roaming dataset.
Data available at: https://junboli-cn.github.io/spags/.
"""

import torch
import json
import numpy as np
import math
import os
from natsort import natsorted

import Framework
from Logging import Logger
from Cameras.Equirectangular import EquirectangularCamera
from Cameras.utils import CameraProperties
from Datasets.Base import BaseDataset
from Datasets.utils import CameraCoordinateSystemsTransformations, loadImagesParallel, WorldCoordinateSystemTransformations, transformPosesPCA
from Datasets.Colmap import quaternion_to_R, storePly, fetchPly, Image, Camera


def angle_axis_to_quaternion(angle_axis: np.ndarray):
    angle = np.linalg.norm(angle_axis)

    x = angle_axis[0] / angle
    y = angle_axis[1] / angle
    z = angle_axis[2] / angle

    qw = math.cos(angle / 2.0)
    qx = x * math.sqrt(1 - qw * qw)
    qy = y * math.sqrt(1 - qw * qw)
    qz = z * math.sqrt(1 - qw * qw)

    return np.array([qw, qx, qy, qz])


def read_points3D_opensfm(reconstructions):
    num_points = 0
    for reconstruction in reconstructions:
        num_points = num_points + len(reconstruction["points"])

    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    for reconstruction in reconstructions:

        for i in (reconstruction["points"]):
            color = (reconstruction["points"][i]["color"])
            coordinates = (reconstruction["points"][i]["coordinates"])
            xyz = np.array([coordinates[0], coordinates[1], coordinates[2]])
            rgb = np.array([color[0], color[1], color[2]])
            error = np.array(0)
            xyzs[count] = xyz
            rgbs[count] = rgb
            errors[count] = error
            count += 1

    return xyzs, rgbs, errors


def read_opensfm(reconstructions, focal_angle):
    images = {}
    i = 0
    cameras = {}
    camera_names = {}
    for reconstruction in reconstructions:
        # read camera intrinsics
        for i, camera in enumerate(reconstruction["cameras"]):
            camera_name = camera
            camera_info = reconstruction["cameras"][camera]
            if camera_info['projection_type'] in ['spherical', 'equirectangular']:
                camera_id = 11
                model = "SPHERICAL"
                width = reconstruction["cameras"][camera]["width"]
                height = reconstruction["cameras"][camera]["height"]
                focal_x = width / (2 * math.pi * math.cos(focal_angle / 180 * math.pi))
                focal_y = focal_x
                center_x = width / 2
                center_y = height / 2
                params = np.array([focal_x, focal_y, center_x, center_y])
                cameras[camera_id] = Camera(id=camera_id, model=model, width=width, height=height, params=params)
                camera_names[camera_name] = camera_id
            else:
                raise NotImplementedError(f"{camera_info['projection_type']} camera model from OpenSfM data format is not implemented")

        # read camera extrinsics
        for shot in reconstruction["shots"]:
            translation = reconstruction["shots"][shot]["translation"]
            rotation = reconstruction["shots"][shot]["rotation"]
            qvec = angle_axis_to_quaternion(rotation)
            tvec = np.array([translation[0], translation[1], translation[2]])
            camera_name = reconstruction["shots"][shot]["camera"]
            camera_id = camera_names.get(camera_name, 0)
            image_id = i
            image_name = shot
            xys = np.array([0, 0]) # dummy
            point3D_ids = np.array([0, 0]) # dummy
            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name, xys=xys, point3D_ids=point3D_ids)
            i += 1
    return cameras, images


@Framework.Configurable.configure(
    PATH='dataset/RaR/pano/O_lion',
    BACKGROUND_COLOR=[0.0, 0.0, 0.0],
    NEAR_PLANE=0.2,
    FAR_PLANE=1000.0,
    TEST_STEP=8,
    FOCAL_ANGLE=0.0,
)
class CustomDataset(BaseDataset):
    """Dataset class for panorama scenes from the Roaming and Rounding dataset."""

    def __init__(self, path: str) -> None:
        super().__init__(
            path,
            EquirectangularCamera(0.2, 1000.0),
            CameraCoordinateSystemsTransformations.LEFT_HAND,
            WorldCoordinateSystemTransformations.XnZY,
        )

    def load(self) -> dict[str, list[CameraProperties] | None]:
        """Loads the dataset into a dict containing lists of CameraProperties for training, evaluation, and testing."""
        # set near and far plane
        self.camera.near_plane = self.NEAR_PLANE
        self.camera.far_plane = self.FAR_PLANE
        # load dataset
        dataset: dict[str, list[CameraProperties]] = {subset: [] for subset in self.subsets}
        # load data
        reconstruction_file = self.dataset_path / 'reconstruction.json'
        with open(reconstruction_file) as f:
            reconstruction = json.load(f)
        cam_intrinsics, cam_extrinsics = read_opensfm(reconstruction, self.FOCAL_ANGLE)
        # create camera properties
        all_views: list[CameraProperties] = []
        for cam_idx, cam_data in enumerate(cam_intrinsics.values()):
            # load images
            images = [data for data in cam_extrinsics.values() if data.camera_id == cam_data.id]
            images = natsorted(images, key=lambda data: data.name)
            image_directory_name = 'images'
            image_scale_factor = self.IMAGE_SCALE_FACTOR
            match self.IMAGE_SCALE_FACTOR:
                case 0.5:
                    image_directory_name = 'images_2'
                    image_scale_factor = None
                case _:
                    pass
            image_filenames = [str(self.dataset_path / image_directory_name / image.name) for image in images]
            rgbs, alphas = loadImagesParallel(image_filenames, image_scale_factor, num_threads=-1, desc=f'camera {cam_data.id}')
            for idx, (image, rgb, alpha) in enumerate(zip(images, rgbs, alphas)):
                # extract c2w matrix
                rotation_matrix = torch.from_numpy(quaternion_to_R(image.qvec)).float()
                translation_vector = torch.from_numpy(image.tvec).float()
                w2c = torch.eye(4, device=torch.device('cpu'))
                w2c[:3, :3] = rotation_matrix
                w2c[:3, 3] = translation_vector
                c2w = torch.linalg.inv(w2c)
                # intrinsics
                focal_x = cam_data.params[0]
                focal_y = cam_data.params[1]
                principal_offset_x = cam_data.params[2] - cam_data.width / 2
                principal_offset_y = cam_data.params[3] - cam_data.height / 2
                if self.IMAGE_SCALE_FACTOR is not None:
                    scale_factor_intrinsics_x = rgb.shape[2] / cam_data.width
                    scale_factor_intrinsics_y = rgb.shape[1] / cam_data.height
                    focal_x *= scale_factor_intrinsics_x
                    focal_y *= scale_factor_intrinsics_y
                    principal_offset_x *= scale_factor_intrinsics_x
                    principal_offset_y *= scale_factor_intrinsics_y

                # create camera properties and subsets
                all_views.append(CameraProperties(
                    width=rgb.shape[2],
                    height=rgb.shape[1],
                    rgb=rgb,
                    alpha=alpha,
                    c2w=c2w,
                    focal_x=focal_x,
                    focal_y=focal_y,
                    principal_offset_x=principal_offset_x,
                    principal_offset_y=principal_offset_y,
                ))

        # load point cloud
        ply_path = self.dataset_path / 'reconstruction.ply'
        if not os.path.exists(ply_path):
            Logger.logInfo('Found new scene. Converting sparse SfM points to .ply format.')
            xyz, rgb, _ = read_points3D_opensfm(reconstruction)
            storePly(ply_path, xyz, rgb)
        try:
            self.point_cloud = fetchPly(ply_path)
        except Exception:
            raise Framework.DatasetError(f'Failed to load SfM point cloud')

        # rotate/scale poses to try aligning ground with xy plane
        c2ws = torch.stack([camera.c2w for camera in all_views])
        c2ws, transformation = transformPosesPCA(c2ws, rescale=False)
        for camera_properties, c2w in zip(all_views, c2ws):
            camera_properties.c2w = c2w
        self.point_cloud.transform(transformation)
        self.world_coordinate_system = None

        # perform test split
        if self.TEST_STEP > 0:
            for i in range(len(all_views)):
                if i % self.TEST_STEP == 0:
                    dataset['test'].append(all_views[i])
                else:
                    dataset['train'].append(all_views[i])
        else:
            dataset['train'] = all_views

        # return the dataset
        return dataset
