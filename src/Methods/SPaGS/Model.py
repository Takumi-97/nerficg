# -- coding: utf-8 --

"""SPaGS/Model.py: Implementation of the model for the SPaGS method."""

import torch

import Framework
from Datasets.Base import BaseDataset
from Datasets.utils import BasicPointCloud
from Cameras.utils import quaternion_to_rotation_matrix
from Logging import Logger
from Methods.Base.Model import BaseModel
from Methods.GaussianSplatting.utils import inverse_sigmoid, rgb_to_sh0, LRDecayPolicy
from Optim.AdamUtils import replace_param_group_data, prune_param_groups, extend_param_groups
from Thirdparty.SimpleKNN import distCUDA2
from CudaUtils.MortonEncoding import morton_encode
from Methods.SPaGS.SPaGSCudaBackend import update_3d_filter


class Gaussians(torch.nn.Module):
    """Stores a set of points with 3D Gaussian extent."""

    GOF_DENSIFICATION_GRAD = True
    GOF_DENSIFICATION_CLONE = True

    def __init__(self, sh_degree: int, pretrained: bool) -> None:
        super().__init__()
        self.active_sh_degree = sh_degree if pretrained else 0
        self.active_sh_bases = (self.active_sh_degree + 1) ** 2
        self.max_sh_degree = sh_degree
        self.register_parameter('_positions', None)
        self.register_parameter('_sh_0', None)
        self.register_parameter('_sh_rest', None)
        self.register_parameter('_scales', None)
        self.register_parameter('_rotations', None)
        self.register_parameter('_opacities', None)
        self.densification_info = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.training_cameras_extent = 1.0
        self.filter_3D = torch.empty(0)
        self.use_3d_filter = False
        self.distance2filter = 0
        # activation functions
        self.scale_activation = torch.nn.Identity() if pretrained else torch.exp
        self.inverse_scale_activation = torch.nn.Identity() if pretrained else torch.log
        self.opacity_activation = torch.nn.Identity() if pretrained else torch.sigmoid
        self.inverse_opacity_activation = torch.nn.Identity() if pretrained else inverse_sigmoid
        self.rotation_activation = torch.nn.Identity() if pretrained else torch.nn.functional.normalize

    @property
    def get_scales(self) -> torch.Tensor:
        """Returns the Gaussians' scales."""
        return self.scale_activation(self._scales)

    @property
    def get_rotations(self) -> torch.Tensor:
        """Returns the Gaussians' rotations as quaternions."""
        return self.rotation_activation(self._rotations)

    @property
    def get_positions(self) -> torch.Tensor:
        """Returns the Gaussians' means."""
        return self._positions

    @property
    def get_sh_0(self) -> torch.Tensor:
        """Returns the Gaussians' 0-th degree SH features."""
        return self._sh_0

    @property
    def get_sh_rest(self) -> torch.Tensor:
        """Returns the Gaussians' SH features beyond the 0-th degree."""
        return self._sh_rest

    @property
    def get_opacities(self) -> torch.Tensor:
        """Returns the Gaussians' opacities."""
        return self.opacity_activation(self._opacities)

    @property
    def get_opacities_with_3D_filter(self) -> torch.Tensor:
        """Returns the Gaussians' opacities with the 3D filter applied."""
        # apply 3D filter
        scales = self.get_scales
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        scales_after_square = scales_square + torch.square(self.filter_3D)
        det2 = scales_after_square.prod(dim=1)
        coef = torch.sqrt(det1 / det2)
        return self.get_opacities * coef[..., None]

    @property
    def get_scales_with_3D_filter(self) -> torch.Tensor:
        """Returns the Gaussians' scales with the 3D filter applied."""
        scales = self.get_scales
        # apply 3D filter
        scales = torch.square(scales) + torch.square(self.filter_3D)
        scales = torch.sqrt(scales)
        return scales

    @property
    def get_densification_info(self) -> torch.Tensor:
        """Returns the current densification info buffers."""
        return self.densification_info

    def setup_3d_filter(self, dataset: 'BaseDataset', dilation: float = 0.2) -> None:
        """Sets up a 3D filter (see https://arxiv.org/abs/2311.16493)."""
        self.use_3d_filter = True
        max_focal = 1.0e-12
        # we do not use the default dataset iterator to avoid copies, thus it is important to not modify anything here
        for camera_properties in dataset.data[dataset.mode]:
            max_focal = max(max_focal, max(camera_properties.focal_x, camera_properties.focal_y))
        # assume max_focal is focal length of the highest resolution camera
        self.distance2filter = dilation ** 0.5 / max_focal
        self.compute_3d_filter(dataset)

    def compute_3d_filter(self, dataset: 'BaseDataset') -> None:
        """Computes the 3D filter."""
        # dilation = 0.2
        # positions = self.get_positions.contiguous()
        # filter_3d = torch.full((positions.shape[0], ), fill_value=torch.finfo(torch.float32).max, device=positions.device, dtype=torch.float32)
        # # we do not use the default dataset iterator to avoid copies, thus it is important to not modify anything here
        # for camera_properties in dataset.data[dataset.mode]:
        #     dataset.camera.setProperties(camera_properties)
        #     positions_cam = dataset.camera.pointsToCameraSpace(positions)
        #     distances = torch.norm(positions_cam, dim=1)
        #     valid = distances >= dataset.camera.near_plane
        #     max_angle = torch.pi * (dataset.camera.properties.height - 1) / (2 * dataset.camera.properties.height)
        #     max_angle = torch.pi / 2 * 0.5
        #     theta = torch.asin(positions_cam[valid][:, 1] / distances[valid]).abs().clamp_max(max_angle)
        #     f = dataset.camera.properties.width / (2.0 * torch.pi * torch.cos(theta))
        #     current_filter3d = distances[valid] * (dilation ** 0.5) / f
        #     filter_3d[valid] = torch.min(filter_3d[valid], current_filter3d)
        # self.filter_3D = filter_3d[:, None]
        # return

        positions = self.get_positions.contiguous()
        filter_3d = torch.full((positions.shape[0], 1), fill_value=torch.finfo(torch.float32).max, device=positions.device, dtype=torch.float32)
        visibility_mask = torch.zeros((positions.shape[0], 1), device=positions.device, dtype=torch.bool)
        # we do not use the default dataset iterator to avoid copies, thus it is important to not modify anything here
        for camera_properties in dataset.data[dataset.mode]:
            dataset.camera.setProperties(camera_properties)
            update_3d_filter(
                dataset.camera,
                positions,
                filter_3d,
                visibility_mask,
                self.distance2filter,
            )

        filter_3d_max = filter_3d[visibility_mask].max()
        filter_3d = torch.where(visibility_mask, filter_3d, filter_3d_max, out=filter_3d)
        self.filter_3D = filter_3d

    def increase_used_sh_degree(self) -> None:
        """Increases the used SH degree."""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            self.active_sh_bases = (self.active_sh_degree + 1) ** 2

    def initialize_from_point_cloud(self, point_cloud: BasicPointCloud, training_cameras_extent: float) -> None:
        """Initializes the model from a point cloud."""
        self.training_cameras_extent = training_cameras_extent
        positions = point_cloud.positions.cuda()
        rgbs = torch.full_like(positions, fill_value=0.5) if point_cloud.colors is None else point_cloud.colors.cuda()
        n_initial_points = positions.shape[0]
        sh_all = torch.zeros((n_initial_points, (self.max_sh_degree + 1) ** 2, 3), dtype=torch.float32, device='cuda')
        sh_all[:, 0] = rgb_to_sh0(rgbs)

        Logger.logInfo(f'Number of points at initialization: {n_initial_points:,}')

        dist2 = distCUDA2(positions).clamp_min(1e-7)
        scales = self.inverse_scale_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rotations = torch.zeros((n_initial_points, 4), dtype=torch.float32, device='cuda')
        rotations[:, 0] = 1.0

        opacities = self.inverse_opacity_activation(torch.full((n_initial_points, 1), fill_value=0.1, dtype=torch.float32, device='cuda'))

        self._positions = torch.nn.Parameter(positions.contiguous())
        self._sh_0 = torch.nn.Parameter(sh_all[:, 0:1].contiguous())
        self._sh_rest = torch.nn.Parameter(sh_all[:, 1:].contiguous())
        self._scales = torch.nn.Parameter(scales.contiguous())
        self._rotations = torch.nn.Parameter(rotations.contiguous())
        self._opacities = torch.nn.Parameter(opacities.contiguous())
        self.reset_densification_info()

    def training_setup(self, training_wrapper, dataset: 'BaseDataset') -> None:
        """Sets up the optimizer."""
        self.percent_dense = training_wrapper.PERCENT_DENSE

        param_groups = [
            {'params': [self._positions], 'lr': training_wrapper.LEARNING_RATE_POSITION_INIT * self.training_cameras_extent, 'name': 'positions'},
            {'params': [self._sh_0], 'lr': training_wrapper.LEARNING_RATE_FEATURE, 'name': 'sh_0'},
            {'params': [self._sh_rest], 'lr': training_wrapper.LEARNING_RATE_FEATURE / 20.0, 'name': 'sh_rest'},
            {'params': [self._opacities], 'lr': training_wrapper.LEARNING_RATE_OPACITY, 'name': 'opacities'},
            {'params': [self._scales], 'lr': training_wrapper.LEARNING_RATE_SCALING, 'name': 'scales'},
            {'params': [self._rotations], 'lr': training_wrapper.LEARNING_RATE_ROTATION, 'name': 'rotations'}
        ]

        try:
            from Thirdparty.Apex import FusedAdam
            # slightly faster than the PyTorch implementation
            self.optimizer = FusedAdam(param_groups, lr=0.0, eps=1e-15, adam_w_mode=False)
            Logger.logInfo('using apex FusedAdam')
        except Framework.ExtensionError:
            Logger.logWarning('apex is not installed -> using the slightly slower PyTorch Adam instead')
            Logger.logWarning('apex can be installed using ./scripts/install.py -e src/Thirdparty/Apex.py')
            self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15, fused=True)

        self.position_lr_scheduler = LRDecayPolicy(
            lr_init=training_wrapper.LEARNING_RATE_POSITION_INIT * self.training_cameras_extent,
            lr_final=training_wrapper.LEARNING_RATE_POSITION_FINAL * self.training_cameras_extent,
            lr_delay_mult=training_wrapper.LEARNING_RATE_POSITION_DELAY_MULT,
            max_steps=training_wrapper.LEARNING_RATE_POSITION_MAX_STEPS)

        if training_wrapper.USE_3D_FILTER:
            self.setup_3d_filter(dataset)

    def update_learning_rate(self, iteration: int) -> None:
        """ Learning rate scheduling per step """
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'positions':
                scheduled_lr = self.position_lr_scheduler(iteration)
                # pruning後のLRリセット中は scheduled_lr と reset_lr を線形補間
                if (hasattr(self, '_pplr_iter_end') and iteration <= self._pplr_iter_end):
                    t = (iteration - self._pplr_iter_start) / max(self._pplr_iter_end - self._pplr_iter_start, 1)
                    t = min(max(t, 0.0), 1.0)
                    lr = self._pplr_start + (scheduled_lr - self._pplr_start) * t
                    param_group['lr'] = max(lr, scheduled_lr)
                else:
                    param_group['lr'] = scheduled_lr

    def set_post_pruning_lr(self, reset_lr: float, start_iter: int, transition_steps: int) -> None:
        """Pruning後に position LR を一時的にリセットし、通常スケジュールへ線形補間する。

        pruning直後は position LR が低くなりすぎていて形状回復できないため、
        reset_lr から scheduled_lr へ transition_steps かけて移行する。
        """
        self._pplr_start = reset_lr
        self._pplr_iter_start = start_iter
        self._pplr_iter_end = start_iter + transition_steps
        Logger.logInfo(f'  Post-pruning LR reset: {reset_lr:.2e} → scheduled over {transition_steps} iters')

    def reset_opacities(self, max_opacity: float) -> None:
        """Resets the opacities to a fixed value."""
        current_opacities = self.get_opacities_with_3D_filter if self.use_3d_filter else self.get_opacities
        opacities_new = current_opacities.clamp_max(max_opacity)
        if self.use_3d_filter:
            # make sure that the current 3d filter has the same effect on the new opacities
            scales_square = torch.square(self.get_scales)
            det1 = scales_square.prod(dim=1)
            scales_after_square = scales_square + torch.square(self.filter_3D)
            det2 = scales_after_square.prod(dim=1)
            coef = torch.sqrt(det1 / det2)
            opacities_new = opacities_new / coef[..., None]
        opacities_new = self.inverse_opacity_activation(opacities_new)
        replace_param_group_data(self.optimizer, opacities_new, 'opacities')

    def decay_opacities(self, decay_factor: float):
        """Decays the opacities by a factor."""
        opacities_new = self.inverse_opacity_activation(self.get_opacities * decay_factor)
        replace_param_group_data(self.optimizer, opacities_new, 'opacities')
        # these lines match the behavior of reset_opacities, but aren't necessary as the decay is only a small change
        # current_opacities = self.get_opacities_with_3D_filter if self.use_3d_filter else self.get_opacities
        # opacities_new = current_opacities * decay_factor
        # if self.use_3d_filter:
        #     # make sure that the current 3d filter has the same effect on the new opacities
        #     scales = self.get_scales
        #     scales_square = torch.square(scales)
        #     det1 = scales_square.prod(dim=1)
        #     scales_after_square = scales_square + torch.square(self.filter_3D)
        #     det2 = scales_after_square.prod(dim=1)
        #     coef = torch.sqrt(det1 / det2)
        #     opacities_new = opacities_new / coef[..., None]
        # opacities_new = self.inverse_opacity_activation(opacities_new)
        # replace_param_group_data(self.optimizer, opacities_new, 'opacities')

    def reduce_opacity(self, amount: float, ignore_3d_filter: bool = True):
        if ignore_3d_filter:
            opacities_new = self.inverse_opacity_activation(self.get_opacities - amount)
            replace_param_group_data(self.optimizer, opacities_new, 'opacities')
        else:
            # these lines match the behavior of reset_opacity
            current_opacities = self.get_opacities_with_3D_filter if self.use_3d_filter else self.get_opacities
            opacities_new = current_opacities - amount
            if self.use_3d_filter:
                # make sure that the current 3d filter has the same effect on the new opacities
                scales_square = torch.square(self.get_scales)
                det1 = scales_square.prod(dim=1)
                scales_after_square = scales_square + torch.square(self.filter_3D)
                det2 = scales_after_square.prod(dim=1)
                coef = torch.sqrt(det1 / det2)
                opacities_new = opacities_new / coef[..., None]
            opacities_new = self.inverse_opacity_activation(opacities_new)
            replace_param_group_data(self.optimizer, opacities_new, 'opacities')

    def prune_points(self, prune_mask: torch.Tensor) -> None:
        """Prunes points that are not visible or too large."""
        valid_mask = ~prune_mask
        optimizable_tensors = prune_param_groups(self.optimizer, valid_mask)

        self._positions = optimizable_tensors['positions']
        self._sh_0 = optimizable_tensors['sh_0']
        self._sh_rest = optimizable_tensors['sh_rest']
        self._opacities = optimizable_tensors['opacities']
        self._scales = optimizable_tensors['scales']
        self._rotations = optimizable_tensors['rotations']

    def densification_postfix(
            self,
            new_positions: torch.Tensor,
            new_sh_0: torch.Tensor,
            new_sh_rest: torch.Tensor,
            new_opacities: torch.Tensor,
            new_scales: torch.Tensor,
            new_rotations: torch.Tensor
    ) -> None:
        """Incorporate the changes from the densification step into the parameter groups."""
        optimizable_tensors = extend_param_groups(self.optimizer, {
            'positions': new_positions,
            'sh_0': new_sh_0,
            'sh_rest': new_sh_rest,
            'opacities': new_opacities,
            'scales': new_scales,
            'rotations': new_rotations
        })
        self._positions = optimizable_tensors['positions']
        self._sh_0 = optimizable_tensors['sh_0']
        self._sh_rest = optimizable_tensors['sh_rest']
        self._opacities = optimizable_tensors['opacities']
        self._scales = optimizable_tensors['scales']
        self._rotations = optimizable_tensors['rotations']

    def reset_densification_info(self):
        n_points = self._positions.shape[0]
        n_floats = 3 if Gaussians.GOF_DENSIFICATION_GRAD else 2
        self.densification_info = torch.zeros((n_floats, n_points, 1), dtype=torch.float32, device='cuda')

    def split(self, grads: torch.Tensor, grad_threshold: float, grads_abs: torch.Tensor | None, grad_abs_threshold: float | None) -> torch.Tensor:
        """Densify by splitting Gaussians that satisfy the gradient condition."""
        n_init_points = self.get_positions.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros(n_init_points, dtype=torch.float32, device='cuda')
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        if grads_abs is not None:
            padded_grad_abs = torch.zeros(n_init_points, dtype=torch.float32, device='cuda')
            padded_grad_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
            selected_pts_mask |= torch.where(padded_grad_abs >= grad_abs_threshold, True, False)
        selected_pts_mask &= torch.max(self.get_scales, dim=1).values > self.percent_dense * self.training_cameras_extent

        stds = self.get_scales[selected_pts_mask].repeat(2, 1)
        samples = torch.normal(mean=0.0, std=stds)
        rots = quaternion_to_rotation_matrix(self._rotations[selected_pts_mask]).repeat(2, 1, 1)
        new_positions = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_positions[selected_pts_mask].repeat(2, 1)
        new_scales = self.inverse_scale_activation(self.get_scales[selected_pts_mask].repeat(2, 1) / 1.6)
        new_rotations = self._rotations[selected_pts_mask].repeat(2, 1)
        new_sh_0 = self._sh_0[selected_pts_mask].repeat(2, 1, 1)
        new_sh_rest = self._sh_rest[selected_pts_mask].repeat(2, 1, 1)
        new_opacities = self._opacities[selected_pts_mask].repeat(2, 1)

        self.densification_postfix(new_positions, new_sh_0, new_sh_rest, new_opacities, new_scales, new_rotations)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(2 * selected_pts_mask.sum().item(), device='cuda', dtype=torch.bool)))
        return prune_filter

    def duplicate(self, grads: torch.Tensor, grad_threshold: float, grads_abs: torch.Tensor | None, grad_abs_threshold: float | None) -> None:
        """Densify by duplicating Gaussians that satisfy the gradient condition."""
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(grads.flatten() >= grad_threshold, True, False)
        if grads_abs is not None:
            selected_pts_mask |= torch.where(grads_abs.flatten() >= grad_abs_threshold, True, False)
        selected_pts_mask &= torch.max(self.get_scales, dim=1).values <= self.percent_dense * self.training_cameras_extent

        if Gaussians.GOF_DENSIFICATION_CLONE:
            # sample a new gaussian instead of fixing position (from gof)
            stds = self.get_scales[selected_pts_mask]
            samples = torch.normal(mean=0.0, std=stds)
            rots = quaternion_to_rotation_matrix(self._rotations[selected_pts_mask])
            new_positions = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_positions[selected_pts_mask]
        else:
            new_positions = self._positions[selected_pts_mask]  # 3dgs

        new_sh_0 = self._sh_0[selected_pts_mask]
        new_sh_rest = self._sh_rest[selected_pts_mask]
        new_opacities = self._opacities[selected_pts_mask]
        new_scales = self._scales[selected_pts_mask]
        new_rotations = self._rotations[selected_pts_mask]

        self.densification_postfix(new_positions, new_sh_0, new_sh_rest, new_opacities, new_scales, new_rotations)

    def densify_and_prune(self, grad_threshold: float, min_opacity: float, prune_large_gaussians: bool,
                          sa_opacity_pruning: bool = False, sa_min_weight: float = 0.10) -> None:
        """Densifies the point cloud and prunes points that are not visible or too large."""
        denominator = self.densification_info[0].clamp_min(1.0)
        grads = self.densification_info[1] / denominator
        grads_abs, grad_abs_threshold = None, None
        if Gaussians.GOF_DENSIFICATION_GRAD:
            grads_abs = self.densification_info[2] / denominator
            ratio = (grads.flatten() >= grad_threshold).float().mean()
            grad_abs_threshold = torch.quantile(grads_abs.flatten(), 1.0 - ratio).item()

        self.duplicate(grads, grad_threshold, grads_abs, grad_abs_threshold)
        prune_mask = self.split(grads, grad_threshold, grads_abs, grad_abs_threshold)

        if sa_opacity_pruning:
            # SA（Solid-Angle）補正 opacity プルーニング。
            # 極付近の Gaussian は多くのピクセルに投影されるため opacity が高くなりやすい。
            # cos(θ) で割ることで「極ほど厳しい opacity 閾値」を適用し、
            # densification で過剰に生成される極 Gaussian を抑制する。
            pos = self.get_positions.detach()
            dist_xz = torch.norm(pos[:, [0, 2]], dim=1).clamp(min=1e-6)
            theta = torch.atan2(pos[:, 1], dist_xz)
            solid_w = torch.cos(theta).clamp(min=sa_min_weight)
            sa_threshold = min_opacity / solid_w  # 極(cos→0)ほど threshold が大きくなる
            prune_mask |= self.get_opacities.flatten() < sa_threshold
        else:
            prune_mask |= self.get_opacities.flatten() < min_opacity
        if prune_large_gaussians:
            prune_mask |= self.get_scales.max(dim=1).values > 0.1 * self.training_cameras_extent
        self.prune_points(prune_mask)

        self.reset_densification_info()

        torch.cuda.empty_cache()

    def importance_pruning(self, max_blending_weights: torch.Tensor, threshold: float) -> None:
        """Prunes points based on the maximum blending weights."""
        mask = max_blending_weights < threshold
        self.prune_points(mask)
        if self.use_3d_filter:
            self.filter_3D = self.filter_3D[~mask].contiguous()

    def bake_activations(self):
        """Bakes relevant activation functions into the final parameters."""
        # bake activation functions into final parameters
        self._rotations.data = self.get_rotations
        self.rotation_activation = torch.nn.Identity()
        # Important: opacities must be baked before scales due to implementation of get_opacities_with_3D_filter
        self._opacities.data = self.get_opacities_with_3D_filter if self.use_3d_filter else self.get_opacities
        self.opacity_activation = torch.nn.Identity()
        self.inverse_opacity_activation = torch.nn.Identity()
        self._scales.data = self.get_scales_with_3D_filter if self.use_3d_filter else self.get_scales
        self.scale_activation = torch.nn.Identity()
        self.inverse_scale_activation = torch.nn.Identity()
        # 3d filter is baked into relevant parameters now
        self.use_3d_filter = False

        # prune points that would never be visible anyway
        self.prune_points((self._opacities < 0.00392156862).squeeze())  # 1/255

        # morton sort
        morton_encoding = morton_encode(self._positions)
        order = torch.argsort(morton_encoding)
        self._positions.data = self._positions[order].contiguous()
        self._rotations.data = self._rotations[order].contiguous()
        self._sh_0.data = self._sh_0[order].contiguous()
        self._sh_rest.data = self._sh_rest[order].contiguous()
        self._scales.data = self._scales[order].contiguous()
        self._opacities.data = self._opacities[order].contiguous()
    
    #以下Claudeより追加
    def reset_contribution_score(self) -> None:
        """球面Contribution scoreの累積バッファをリセット。"""
        n = self._positions.shape[0]
        self.contribution_score = torch.zeros(n, dtype=torch.float32, device='cuda')

    def accumulate_spherical_contribution(
        self,
        weighted_scores: torch.Tensor,  # shape: [N_gaussians]  各Gaussianへの加重スコア
    ) -> None:
        """1フレーム分の球面Contribution scoreを累積する。"""
        if not hasattr(self, 'contribution_score') or self.contribution_score.shape[0] != self._positions.shape[0]:
            self.reset_contribution_score()
        self.contribution_score += weighted_scores

    def spherical_contribution_pruning(
        self,
        keep_ratio: float = 0.5,
    ) -> None:
        """球面Contribution scoreの低いGaussianをprune。
        
        Args:
            keep_ratio: 残すGaussianの割合。0.5 なら上位50%を残す。
        """
        if not hasattr(self, 'contribution_score') or self.contribution_score.shape[0] != self._positions.shape[0]:
            raise RuntimeError('contribution_score が初期化されていません。先に accumulate_spherical_contribution() を呼んでください。')

        scores = self.contribution_score
        threshold = torch.quantile(scores, 1.0 - keep_ratio)
        prune_mask = scores < threshold

        n_before = self._positions.shape[0]
        self.prune_points(prune_mask)
        n_after = self._positions.shape[0]

        if self.use_3d_filter:
            self.filter_3D = self.filter_3D[~prune_mask].contiguous()

        # バッファをリセット
        self.reset_contribution_score()
        torch.cuda.empty_cache()

        Logger.logInfo(f'  Spherical contribution pruning: {n_before:,} → {n_after:,} Gaussians ({100*(1-n_after/n_before):.1f}% removed)')

    def region_aware_pruning(
        self,
        scores: torch.Tensor,
        polar_threshold_deg: float = 45.0,
        keep_near_eq: float = 0.80,
        keep_near_polar: float = 0.50,
        keep_far: float = 0.40,
        far_percentile: float = 0.70,
        hf_scale_percentile: float = 0.40,
        hf_keep_boost: float = 0.12,
        target_keep_ratio: float | None = None,
        pixel_grad_scores: torch.Tensor | None = None,
    ) -> None:
        """3領域 × HF/LF の6サブグループで keep_ratio を変えてGaussianをprune。

        領域定義（Y軸上向き座標系）:
          近景赤道: |仰角| ≤ polar_threshold かつ 距離 ≤ far_percentile分位
          近景極  : |仰角| > polar_threshold かつ 距離 ≤ far_percentile分位
          遠景    : 距離 > far_percentile分位（緯度問わず）

        各領域をさらに HF（スケール下位 hf_scale_percentile）と LF に分け、
        HF は keep_ratio + hf_keep_boost で保護する。

        target_keep_ratio が指定された場合、その値を達成するように各領域の
        keep_ratio を比例スケールする（領域間の優先度比は維持）。
        """
        pos = self._positions.detach()  # [N, 3]
        n_before = pos.shape[0]

        # 仰角 θ（Y上向き想定）
        dist_xz = torch.norm(pos[:, [0, 2]], dim=1).clamp(min=1e-6)
        theta = torch.atan2(pos[:, 1], dist_xz)               # [-π/2, π/2]
        is_polar = theta.abs() > (polar_threshold_deg * torch.pi / 180.0)

        # 遠景判定（シーン中心からの距離の far_percentile 以上）
        scene_center = pos.mean(dim=0)
        dist = torch.norm(pos - scene_center, dim=1)
        far_thr = torch.quantile(dist, far_percentile)
        is_far = dist >= far_thr

        # HF/LF判定 - 近景はグローバルスケール、遠景は遠景内ローカルパーセンタイル
        # 遠景Gaussianはスケールが大きくグローバル閾値では全てLF判定される問題を修正
        # → 遠景の絵画・ドア等の細部Gaussianを正しくHFとして保護
        max_scale = self.get_scales.detach().max(dim=1).values  # [N]
        hf_thr    = torch.quantile(max_scale, hf_scale_percentile)
        is_hf     = max_scale < hf_thr  # 近景: グローバルスケールベース

        # 遠景のみ: 遠景内ローカル閾値で上書き（HF 50% / LF 50% を遠景内で均等化）
        if is_far.any():
            far_hf_thr = torch.quantile(max_scale[is_far], hf_scale_percentile)
            is_hf[is_far] = max_scale[is_far] < far_hf_thr

        if pixel_grad_scores is not None:
            grad_thr_far = torch.quantile(pixel_grad_scores, 1.0 - hf_scale_percentile)
            is_hf_grad   = pixel_grad_scores > grad_thr_far
            is_hf = torch.where(is_far, is_hf_grad, is_hf)

        # target_keep_ratio: HFブーストを考慮した実際の保持率でスケール計算
        if target_keep_ratio is not None:
            n_near_eq       = (~is_polar & ~is_far).sum().item()
            n_near_polar    = ( is_polar & ~is_far).sum().item()
            n_far           = is_far.sum().item()
            n_near_eq_hf    = (~is_polar & ~is_far & is_hf).sum().item()
            n_near_polar_hf = ( is_polar & ~is_far & is_hf).sum().item()
            n_far_hf        = ( is_far            & is_hf).sum().item()
            current_keep = (
                n_near_eq_hf    * min(keep_near_eq    + hf_keep_boost, 0.95)
                + (n_near_eq    - n_near_eq_hf)    * keep_near_eq
                + n_near_polar_hf * min(keep_near_polar + hf_keep_boost, 0.95)
                + (n_near_polar - n_near_polar_hf) * keep_near_polar
                + n_far_hf        * min(keep_far        + hf_keep_boost, 0.95)
                + (n_far        - n_far_hf)        * keep_far
            ) / max(n_before, 1)
            if current_keep > 1e-6:
                scale = target_keep_ratio / current_keep
                keep_near_eq    = min(keep_near_eq    * scale, 0.95)
                keep_near_polar = min(keep_near_polar * scale, 0.95)
                keep_far        = min(keep_far        * scale, 0.95)
                Logger.logInfo(
                    f'  target_keep={target_keep_ratio:.3f}, scale={scale:.3f}'
                    f' → keep_eq={keep_near_eq:.3f}, polar={keep_near_polar:.3f}, far={keep_far:.3f}'
                )

        prune_mask = torch.zeros(n_before, dtype=torch.bool, device=pos.device)

        for region_mask, keep_ratio, label in [
            (~is_polar & ~is_far, keep_near_eq,    '近景赤道'),
            ( is_polar & ~is_far, keep_near_polar, '近景極  '),
            ( is_far,             keep_far,        '遠景    '),
        ]:
            for sub_mask, sub_keep, sub_tag in [
                (region_mask & is_hf,  min(keep_ratio + hf_keep_boost, 0.95), 'HF'),
                (region_mask & ~is_hf, keep_ratio,                            'LF'),
            ]:
                n_sub = sub_mask.sum().item()
                if n_sub < 2:
                    continue
                thr = torch.quantile(scores[sub_mask], 1.0 - sub_keep)
                prune_mask[sub_mask] = scores[sub_mask] < thr
                n_kept = int(n_sub - prune_mask[sub_mask].sum().item())
                Logger.logInfo(f'    {label}[{sub_tag}]: {n_sub:,} → {n_kept:,} kept ({sub_keep*100:.0f}%)')

        self.prune_points(prune_mask)
        if self.use_3d_filter:
            self.filter_3D = self.filter_3D[~prune_mask].contiguous()
        self.reset_contribution_score()
        torch.cuda.empty_cache()

        n_after = self._positions.shape[0]
        Logger.logInfo(f'  Region-aware pruning: {n_before:,} → {n_after:,} ({100*(1-n_after/n_before):.1f}% removed)')

    def opacity_pruning(self, keep_ratio: float) -> None:
        """Uniformly prune by opacity: keep top keep_ratio fraction."""
        n_before = self._positions.shape[0]
        opacities = self.get_opacities.squeeze().detach()
        n_keep = max(1, int(n_before * keep_ratio))
        threshold = torch.topk(opacities, n_keep, largest=True).values[-1]
        prune_mask = opacities < threshold
        self.prune_points(prune_mask)
        if self.use_3d_filter:
            self.filter_3D = self.filter_3D[~prune_mask].contiguous()
        self.reset_contribution_score()
        torch.cuda.empty_cache()
        n_after = self._positions.shape[0]
        Logger.logInfo(f'  Opacity pruning: {n_before:,} → {n_after:,} ({100*(1-n_after/n_before):.1f}% removed)')

    def uniform_score_pruning(self, scores: torch.Tensor, keep_ratio: float) -> None:
        """Uniformly prune by given scores: keep top keep_ratio fraction globally."""
        n_before = scores.shape[0]
        n_keep = max(1, int(n_before * keep_ratio))
        sorted_indices = torch.argsort(scores, descending=True)
        prune_mask = torch.ones(n_before, dtype=torch.bool, device=scores.device)
        prune_mask[sorted_indices[:n_keep]] = False
        self.prune_points(prune_mask)
        if self.use_3d_filter:
            self.filter_3D = self.filter_3D[~prune_mask].contiguous()
        self.reset_contribution_score()
        torch.cuda.empty_cache()
        n_after = self._positions.shape[0]
        Logger.logInfo(f'  Uniform score pruning: {n_before:,} → {n_after:,} ({100*(1-n_after/n_before):.1f}% removed)')

    def region_normalized_pruning(
        self,
        scores: torch.Tensor,
        keep_ratio: float,
        polar_threshold_deg: float = 45.0,
        far_percentile: float = 0.70,
    ) -> None:
        """360°固有のバイアスを補正した貢献スコアプルーニング。

        問題：遠景 Gaussian はカメラから遠いためブレンド重みが小さく、
              同等の重要度でも近景より絶対スコアが低くなる「距離バイアス」がある。
              グローバルな閾値を使うと遠景が過剰に削除される。

        解決：3つの region（近景赤道 / 近景極 / 遠景）内でそれぞれ z-score 正規化し、
              「その region の中での相対的な重要度」に変換してからグローバル top-K を行う。
              これにより各 region からバランスよく Gaussian が保持される。

        注意：top-K を threshold 比較で行うと、スコア=0 の大量 Gaussian で tie が発生し
              意図した keep 数を超える場合がある。argsort で厳密に n_keep 個を選ぶ。
        """
        import math
        pos = self._positions.detach()  # [N, 3]
        n_before = pos.shape[0]

        # region割り当て（region_aware_pruningと同じ定義）
        dist_xz = torch.norm(pos[:, [0, 2]], dim=1).clamp(min=1e-6)
        theta = torch.atan2(pos[:, 1], dist_xz)
        is_polar = theta.abs() > (polar_threshold_deg * math.pi / 180.0)

        scene_center = pos.mean(dim=0)
        dist = torch.norm(pos - scene_center, dim=1)
        far_thr = torch.quantile(dist, far_percentile)
        is_far = dist >= far_thr

        regions = [
            (~is_polar & ~is_far, '近景赤道'),
            ( is_polar & ~is_far, '近景極  '),
            ( is_far,             '遠景    '),
        ]

        # 各region内でz-score正規化
        normalized_scores = scores.clone().float()
        for mask, label in regions:
            n_region = mask.sum().item()
            if n_region < 2:
                continue
            s = scores[mask].float()
            mean = s.mean()
            std = s.std().clamp(min=1e-8)
            normalized_scores[mask] = (s - mean) / std
            Logger.logInfo(f'    {label}: n={n_region:,}, score mean={mean:.4f}, std={std:.4f}')

        # グローバルtop-K（正規化スコアで厳密にn_keep個を保持）
        # threshold比較だとスコアが0の大量Gaussianでtieが発生し保持数が過多になるため
        # argsortで順位付けして確実にn_keep個を選ぶ
        n_keep = max(1, int(n_before * keep_ratio))
        sorted_indices = torch.argsort(normalized_scores, descending=True)
        prune_mask = torch.ones(n_before, dtype=torch.bool, device=normalized_scores.device)
        prune_mask[sorted_indices[:n_keep]] = False

        self.prune_points(prune_mask)
        if self.use_3d_filter:
            self.filter_3D = self.filter_3D[~prune_mask].contiguous()
        self.reset_contribution_score()
        torch.cuda.empty_cache()
        n_after = self._positions.shape[0]
        Logger.logInfo(f'  Region-normalized pruning: {n_before:,} → {n_after:,} ({100*(1-n_after/n_before):.1f}% removed)')

@Framework.Configurable.configure(
    SH_DEGREE=3,
)
class SPaGSModel(BaseModel):
    """Defines the SPaGS model."""

    def __init__(self, name: str = None) -> None:
        super().__init__(name)
        self.gaussians: Gaussians | None = None

    def build(self) -> 'SPaGSModel':
        """Builds the model."""
        pretrained = self.num_iterations_trained > 0
        self.gaussians = Gaussians(self.SH_DEGREE, pretrained)
        return self