# -- coding: utf-8 --

"""SPaGS/Renderer.py: """

import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.Base import BaseDataset
from Logging import Logger
from Methods.SPaGS.SPaGSCudaBackend import SPaGSRasterizer
from Methods.Base.Renderer import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.SPaGS.Model import SPaGSModel
from Visual.utils import pseudoColorDepth
from Cameras.utils import transformPoints


@Framework.Configurable.configure(
    BLEND_MODE=0,
    K=16,
    SCALE_MODIFIER=1.0,
    DISABLE_SH0=False,
    DISABLE_SH1=False,
    DISABLE_SH2=False,
    DISABLE_SH3=False,
    USE_MEDIAN_DEPTH=False,
    FORCE_OPTIMIZED_INFERENCE=False,
)
class SPaGSRenderer(BaseRenderer):
    """Renderer for the SPaGS method."""

    def __init__(self, model: 'BaseModel') -> None:
        super().__init__(model, [SPaGSModel])
        if not Framework.config.GLOBAL.GPU_INDICES:
            raise Framework.RendererError('renderer not implemented in CPU mode')
        if len(Framework.config.GLOBAL.GPU_INDICES) > 1:
            Logger.logWarning(f'renderer not implemented in multi-GPU mode: using GPU {Framework.config.GLOBAL.GPU_INDICES[0]}')
        self.rasterizer = SPaGSRasterizer()
        if self.BLEND_MODE not in [0, 1, 2, 3]:
            raise Framework.RendererError('Invalid blend mode')
        if self.BLEND_MODE < 2 and self.K not in [1, 2, 4, 8, 16, 32]:
            Logger.logWarning(f'unsupported K value for selected blend mode may lead to undefined behavior: {self.K}')

    def renderImage(self, camera: 'PerspectiveCamera', to_chw: bool = False, benchmark: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given camera."""
        if benchmark or self.FORCE_OPTIMIZED_INFERENCE:
            return self.renderImageBenchmark(camera, to_chw=to_chw or benchmark)
        else:
            return self.renderImageInference(camera, to_chw)

    def renderImageTraining(self, camera: 'PerspectiveCamera', update_densification_info: bool, use_distance_scaling: bool) -> torch.Tensor:
        """Renders an image for a given camera for optimization."""
        return self.rasterizer(
            positions=self.model.gaussians.get_positions,
            scales=self.model.gaussians.get_scales_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_scales,
            rotations=self.model.gaussians.get_rotations,
            opacities=self.model.gaussians.get_opacities_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_opacities,
            sh_0=self.model.gaussians.get_sh_0,
            sh_rest=self.model.gaussians.get_sh_rest,
            densification_info=self.model.gaussians.get_densification_info if update_densification_info else torch.empty(0),
            camera=camera,
            mode=self.BLEND_MODE,
            K=self.K,
            active_sh_bases=self.model.gaussians.active_sh_bases,
            scale_modifier=1.0,
            use_distance_scaling=use_distance_scaling and update_densification_info,
        )

    @torch.no_grad()
    def renderImageInference(self, camera: 'PerspectiveCamera', to_chw: bool) -> dict[str, torch.Tensor]:
        """Renders an image for a given camera during inference."""
        # modify sh features for visualization
        sh_0 = self.model.gaussians.get_sh_0
        if self.DISABLE_SH0:
            sh_0 = torch.zeros_like(sh_0)
        sh_rest = self.model.gaussians.get_sh_rest
        if self.DISABLE_SH1 or self.DISABLE_SH2 or self.DISABLE_SH3:
            sh_rest = sh_rest.clone()
        if self.DISABLE_SH1:
            sh_rest[:, 0:3].zero_()
        if self.DISABLE_SH2:
            sh_rest[:, 3:8].zero_()
        if self.DISABLE_SH3:
            sh_rest[:, 8:15].zero_()

        rgb, depth = self.rasterizer.render(
            positions=self.model.gaussians.get_positions,
            scales=self.model.gaussians.get_scales_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_scales,
            rotations=self.model.gaussians.get_rotations,
            opacities=self.model.gaussians.get_opacities_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_opacities,
            sh_0=sh_0,
            sh_rest=sh_rest,
            camera=camera,
            mode=self.BLEND_MODE,
            K=self.K,
            active_sh_bases=self.model.gaussians.active_sh_bases,
            scale_modifier=self.SCALE_MODIFIER,
            to_chw=to_chw,
            use_median_depth=self.USE_MEDIAN_DEPTH,
        )
        return {
            'rgb': rgb,
            'depth': depth,
        }

    @torch.inference_mode()
    def renderImageBenchmark(self, camera: 'PerspectiveCamera', to_chw: bool) -> dict[str, torch.Tensor]:
        """Renders an image for a given camera."""
        rgb = self.rasterizer.benchmark(
            positions=self.model.gaussians.get_positions,
            scales=self.model.gaussians.get_scales_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_scales,
            rotations=self.model.gaussians.get_rotations,
            opacities=self.model.gaussians.get_opacities_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_opacities,
            sh_0=self.model.gaussians.get_sh_0,
            sh_rest=self.model.gaussians.get_sh_rest,
            camera=camera,
            mode=self.BLEND_MODE,
            K=self.K,
            active_sh_bases=self.model.gaussians.active_sh_bases,
            scale_modifier=self.SCALE_MODIFIER,
            to_chw=to_chw,
        )
        return { 'rgb': rgb }

    def computeMaxWeights(self, dataset: BaseDataset, threshold: float) -> torch.Tensor:
        """Computes the maximum blending weights for the current dataset."""
        positions = self.model.gaussians.get_positions
        scales = self.model.gaussians.get_scales_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_scales
        opacities = self.model.gaussians.get_opacities_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_opacities
        rotations = self.model.gaussians.get_rotations
        max_weights = torch.zeros(opacities.shape[0], device=opacities.device, dtype=opacities.dtype)
        # we do not use the default dataset iterator to avoid copies, thus it is important to not modify anything here
        for camera_properties in dataset.data[dataset.mode]:
            dataset.camera.setProperties(camera_properties)
            self.rasterizer.update_max_weights(
                max_weights=max_weights,
                positions=positions,
                scales=scales,
                rotations=rotations,
                opacities=opacities,
                camera=dataset.camera,
                mode=self.BLEND_MODE,
                K=self.K,
                active_sh_bases=0,
                scale_modifier=1.0,
                weight_threshold=threshold,
            )
        return max_weights
    ##このDEF追加
    # def computeSphericalContributionScores(
    #         self,
    #         dataset: 'BaseDataset',
    #         n_sample_frames: int = 200,
    #     ) -> torch.Tensor:
    #         """球面Contribution scoreを計算して返す。"""
    #         import random
    #         import math

    #         n_gaussians = self.model.gaussians.get_opacities.shape[0]
    #         scores = torch.zeros(n_gaussians, device='cuda', dtype=torch.float32)

    #         all_camera_props = list(dataset.data[dataset.mode])
    #         sampled = random.sample(all_camera_props, min(n_sample_frames, len(all_camera_props)))

    #         for camera_properties in sampled:
    #             dataset.camera.setProperties(camera_properties)

    #             # 既存の computeMaxWeights と同じCUDA呼び出しを使う（安全）
    #             positions = self.model.gaussians.get_positions
    #             scales    = self.model.gaussians.get_scales_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_scales
    #             opacities = self.model.gaussians.get_opacities_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_opacities
    #             rotations = self.model.gaussians.get_rotations

    #             frame_weights = torch.zeros(n_gaussians, device='cuda', dtype=torch.float32)
    #             self.rasterizer.update_max_weights(
    #                 max_weights=frame_weights,
    #                 positions=positions,
    #                 scales=scales,
    #                 rotations=rotations,
    #                 opacities=opacities,
    #                 camera=dataset.camera,
    #                 mode=self.BLEND_MODE,
    #                 K=self.K,
    #                 active_sh_bases=0,
    #                 scale_modifier=1.0,
    #                 weight_threshold=0.01,
    #             )#wi=Ti*ai,Ti=Π(1-aj)前にあるガウスにどれだけさえぎられているのか

    #             # 立体角補正: カメラ位置をCPUで取得してPythonで計算
    #             T = camera_properties.T  # カメラ中心 [3]
    #             if isinstance(T, torch.Tensor):
    #                 cam_pos = T.float().cpu()
    #             else:
    #                 cam_pos = torch.tensor(T, dtype=torch.float32)

    #             # ワールド座標でGaussian→カメラ方向を計算（すべてCPU→CUDA）
    #             cam_pos_cuda = cam_pos.to('cuda')
    #             dirs = positions.detach() - cam_pos_cuda.unsqueeze(0)  # [N, 3]　カメラ方向ベクトルx-c
    #             dist = torch.norm(dirs, dim=1).clamp(min=1e-6)#正規化

    #             # 仰角 θ（y軸を上方向とする）
    #             theta = torch.acos((dirs[:, 1] / dist).clamp(-1.0, 1.0))#arccos(d/r)
    #             solid_angle_weight = torch.sin(theta).clamp(min=1e-6)

    #             scores += frame_weights * solid_angle_weight

    #         scores /= max(len(sampled), 1)#score=1/V sigma(w*sin theta)
    #         return scores
    
    # def computeSphericalContributionScores(
    #     self,
    #     dataset: 'BaseDataset',
    #     n_sample_frames: int = 200,
    #     beta: float = 0.5,
    # ) -> torch.Tensor:
    #     """論文式(10)(11)に基づくContribution score計算。
        
    #     S_i = Σ_r α_i(r)・T_i(r)・γ(Σ_i)
    #     γ(Σ_i) = V_norm^β,  V_norm = min(V(Σ_i) / V_max90, 1)
    #     """
    #     import random
    #     import math

    #     n_gaussians = self.model.gaussians.get_opacities.shape[0]

    #     # ── 1. 体積項 γ(Σ) を事前計算（全Gaussianで1回） ──
    #     # Gaussianの体積 ∝ det(Σ)^(1/2) = s0*s1*s2（スケール3軸の積）
    #     scales = (
    #         self.model.gaussians.get_scales_with_3D_filter
    #         if self.model.gaussians.use_3d_filter
    #         else self.model.gaussians.get_scales
    #     ).detach()  # [N, 3]

    #     volume = scales[:, 0] * scales[:, 1] * scales[:, 2]  # [N]

    #     # 上位90%タイルで正規化してclamp
    #     v_max90 = torch.quantile(volume, 0.90).clamp(min=1e-9)
    #     v_norm = (volume / v_max90).clamp(max=1.0)           # [N], ∈[0,1]
    #     gamma = v_norm.pow(beta)                              # [N]  γ(Σ)=V_norm^β

    #     # ── 2. 全フレームを通じたブレンド重みの累積 ──
    #     # update_max_weights は最大値しか返さないため、
    #     # 全フレームのmax_weightsの平均を累積重みの代替とする。
    #     # （CUDAバックエンドが累積重みAPIを持たないため最良の近似）
    #     scores_sum = torch.zeros(n_gaussians, device='cuda', dtype=torch.float32)

    #     positions = self.model.gaussians.get_positions.detach()
    #     opacities = (
    #         self.model.gaussians.get_opacities_with_3D_filter
    #         if self.model.gaussians.use_3d_filter
    #         else self.model.gaussians.get_opacities
    #     ).detach()
    #     rotations = self.model.gaussians.get_rotations.detach()
    #     scales_render = (
    #         self.model.gaussians.get_scales_with_3D_filter
    #         if self.model.gaussians.use_3d_filter
    #         else self.model.gaussians.get_scales
    #     ).detach()

    #     all_camera_props = list(dataset.data[dataset.mode])
    #     sampled = random.sample(all_camera_props, min(n_sample_frames, len(all_camera_props)))

    #     for camera_properties in sampled:
    #         dataset.camera.setProperties(camera_properties)

    #         # α_i・T_i の近似：update_max_weights でフレームごとの最大ブレンド重みを取得
    #         frame_weights = torch.zeros(n_gaussians, device='cuda', dtype=torch.float32)
    #         self.rasterizer.update_max_weights(
    #             max_weights=frame_weights,
    #             positions=positions,
    #             scales=scales_render,
    #             rotations=rotations,
    #             opacities=opacities,
    #             camera=dataset.camera,
    #             mode=self.BLEND_MODE,
    #             K=self.K,
    #             active_sh_bases=0,
    #             scale_modifier=1.0,
    #             weight_threshold=0.01,
    #         )

    #         # S_k = Σ_r w_i(r) · γ(Σ_i)  ← 式(10)
    #         scores_sum += frame_weights * gamma

    #     # ── 3. 式(11): フレーム平均 ──
    #     scores = scores_sum / max(len(sampled), 1)
    #     return scores
    
    def computeSphericalContributionScores(
            self,
            dataset: 'BaseDataset',
            n_sample_frames: int = 200,
            beta: float = 0.5,
            gamma_min: float = 0.50,
            hf_frame_boost: float = 5.0,
            use_volume: bool = True,
            use_spherical: bool = True,
            use_distance: bool = False,
            distance_lambda: float = 1.0,
            compute_pixel_grad: bool = False,
            use_error_weight: bool = False,
            error_lambda: float = 2.0,
        ) -> torch.Tensor:
            """360°対応 Contribution score（HFフレーム重み付き・フレーム平均版）。

            S_i = weighted_avg_weight_i · γ(Σ_i) · f(θ_i) · g(d_i)

            weighted_avg_weight_i: GTのグラジエント強度でフレームを重み付けした平均ブレンド重み。
                          高周波（テクスチャ・エッジ豊富）なフレームへの寄与を高く評価することで、
                          高周波成分を担うGaussianが低スコアになるバイアスを補正する。
                          hf_frame_boost=0 で従来の単純平均と等価。
            γ(Σ_i) = clamp(V_norm^β, min=gamma_min)
                          体積補正。gamma_min > 0 で最小Gaussian（高周波成分）の過剰ペナルティを緩和。
            f(θ_i) = cos(θ_i)          仰角補正（極付近を低スコアに）
            g(d_i) = exp(-λ · d_norm_i) 距離補正（use_distance=True時のみ）
            """
            import random
            import math

            n_gaussians = self.model.gaussians.get_opacities.shape[0]

            positions = self.model.gaussians.get_positions.detach()  # [N, 3]
            scales = (
                self.model.gaussians.get_scales_with_3D_filter
                if self.model.gaussians.use_3d_filter
                else self.model.gaussians.get_scales
            ).detach()
            opacities = (
                self.model.gaussians.get_opacities_with_3D_filter
                if self.model.gaussians.use_3d_filter
                else self.model.gaussians.get_opacities
            ).detach()
            rotations = self.model.gaussians.get_rotations.detach()

            # ── 1. 体積項 γ(Σ) ──
            if use_volume:
                volume  = scales[:, 0] * scales[:, 1] * scales[:, 2]
                v_max90 = torch.quantile(volume, 0.90).clamp(min=1e-9)
                v_norm  = (volume / v_max90).clamp(max=1.0)
                gamma   = v_norm.pow(beta).clamp(min=gamma_min)
            else:
                gamma = torch.ones(n_gaussians, device=positions.device)

            # ── 2. 仰角補正 f(θ) = cos(θ) ──
            if use_spherical:
                dist_xz     = torch.norm(positions[:, [0, 2]], dim=1).clamp(min=1e-6)
                theta       = torch.atan2(positions[:, 1], dist_xz)   # [-π/2, π/2]
                solid_angle = torch.cos(theta).clamp(min=0.1)         # 極で0.1下限
            else:
                solid_angle = torch.ones(n_gaussians, device=positions.device)

            # ── 3. 距離補正 g(d) = exp(-λ · d_norm) ──
            if use_distance:
                scene_center = positions.mean(dim=0)
                dist         = torch.norm(positions - scene_center, dim=1)
                d_max90      = torch.quantile(dist, 0.90).clamp(min=1e-6)
                d_norm       = (dist / d_max90).clamp(max=1.0)
                dist_weight  = torch.exp(-distance_lambda * d_norm)
            else:
                dist_weight = torch.ones(n_gaussians, device=positions.device)

            # ── 4. HFフレーム重み付き平均ブレンド重み ──
            # フレームごとのGTグラジエント強度でフレームを重み付けして累積する。
            # テクスチャ・エッジが多いフレーム（高周波）ほど重みを大きくすることで、
            # 高周波成分を担うGaussianのスコアが不当に低くなるバイアスを補正する。
            all_camera_props = list(dataset.data[dataset.mode])
            sampled = random.sample(all_camera_props, min(n_sample_frames, len(all_camera_props)))

            scores_sum = torch.zeros(n_gaussians, device=positions.device, dtype=torch.float32)
            if compute_pixel_grad:
                grad_scores_sum = torch.zeros(n_gaussians, device=positions.device, dtype=torch.float32)
                n_grad_frames = 0
            if use_error_weight:
                error_weighted_sum = torch.zeros(n_gaussians, device=positions.device, dtype=torch.float32)
                error_weight_total = torch.zeros(n_gaussians, device=positions.device, dtype=torch.float32)
                sh_0 = self.model.gaussians.get_sh_0.detach()
                sh_rest = self.model.gaussians.get_sh_rest.detach()
                active_sh_bases = self.model.gaussians.active_sh_bases
            for camera_properties in sampled:
                dataset.camera.setProperties(camera_properties)
                frame_weights = torch.zeros(n_gaussians, device=positions.device, dtype=torch.float32)
                self.rasterizer.update_max_weights(
                    max_weights=frame_weights,
                    positions=positions,
                    scales=scales,
                    rotations=rotations,
                    opacities=opacities,
                    camera=dataset.camera,
                    mode=self.BLEND_MODE,
                    K=self.K,
                    active_sh_bases=0,
                    scale_modifier=1.0,
                    weight_threshold=0.01,
                )
                if hf_frame_boost > 0.0 and camera_properties.rgb is not None:
                    gt = camera_properties.rgb.detach().float()  # [3, H, W]
                    H, W = gt.shape[-2], gt.shape[-1]
                    gx = (gt[:, :, 1:] - gt[:, :, :-1]).abs().mean(dim=0)  # [H, W-1]
                    gy = (gt[:, 1:, :] - gt[:, :-1, :]).abs().mean(dim=0)  # [H-1, W]
                    grad_map = torch.zeros(H, W, device=positions.device)
                    grad_map[:, :W - 1] += gx.to(positions.device)
                    grad_map[:H - 1, :] += gy.to(positions.device)
                    T = camera_properties.T
                    if isinstance(T, torch.Tensor):
                        cam_pos = T.float().to(positions.device)
                    else:
                        cam_pos = torch.tensor(T, dtype=torch.float32, device=positions.device)
                    dirs = positions - cam_pos.unsqueeze(0)  # [N, 3]
                    dist_xz = torch.norm(dirs[:, [0, 2]], dim=1).clamp(min=1e-6)
                    theta = torch.atan2(dirs[:, 1], dist_xz)         # elevation [-pi/2, pi/2]
                    phi = torch.atan2(dirs[:, 0], dirs[:, 2])         # azimuth   [-pi, pi]
                    row = ((0.5 - theta / math.pi) * H).long().clamp(0, H - 1)
                    col = ((0.5 + phi / (2.0 * math.pi)) * W).long().clamp(0, W - 1)
                    grad_at_pos = grad_map[row, col]                  # [N]
                    hf_multiplier = 1.0 + hf_frame_boost * grad_at_pos
                    scores_sum += frame_weights * hf_multiplier
                else:
                    scores_sum += frame_weights

                # ── エラー重み付き累積 ──
                if use_error_weight and camera_properties.rgb is not None:
                    gt = camera_properties.rgb.detach().float().to(positions.device)  # [3, H, W]
                    rgb_render, _ = self.rasterizer.render(
                        positions=positions, scales=scales, rotations=rotations, opacities=opacities,
                        sh_0=sh_0, sh_rest=sh_rest, camera=dataset.camera,
                        mode=self.BLEND_MODE, K=self.K, active_sh_bases=active_sh_bases,
                        scale_modifier=1.0, to_chw=True, use_median_depth=False,
                    )
                    error_map = (rgb_render.detach() - gt).abs().mean(dim=0)  # [H, W]
                    H_e, W_e = error_map.shape
                    T = camera_properties.T
                    if isinstance(T, torch.Tensor):
                        cam_pos = T.float().to(positions.device)
                    else:
                        cam_pos = torch.tensor(T, dtype=torch.float32, device=positions.device)
                    dirs = positions - cam_pos.unsqueeze(0)
                    dist_xz_e = torch.norm(dirs[:, [0, 2]], dim=1).clamp(min=1e-6)
                    theta_e = torch.atan2(dirs[:, 1], dist_xz_e)
                    phi_e = torch.atan2(dirs[:, 0], dirs[:, 2])
                    row_e = ((0.5 - theta_e / math.pi) * H_e).long().clamp(0, H_e - 1)
                    col_e = ((0.5 + phi_e / (2.0 * math.pi)) * W_e).long().clamp(0, W_e - 1)
                    error_at_pos = error_map[row_e, col_e]  # [N]
                    error_weighted_sum += frame_weights * error_at_pos
                    error_weight_total += frame_weights

                if compute_pixel_grad and camera_properties.rgb is not None:
                    gt = camera_properties.rgb.detach().float()  # [3, H, W]
                    H_g, W_g = gt.shape[-2], gt.shape[-1]
                    gx = (gt[:, :, 1:] - gt[:, :, :-1]).abs().mean(dim=0)
                    gy = (gt[:, 1:, :] - gt[:, :-1, :]).abs().mean(dim=0)
                    grad_map = torch.zeros(H_g, W_g, device=positions.device)
                    grad_map[:, :W_g - 1] += gx.to(positions.device)
                    grad_map[:H_g - 1, :] += gy.to(positions.device)
                    T = camera_properties.T
                    if isinstance(T, torch.Tensor):
                        cam_pos = T.float().to(positions.device)
                    else:
                        cam_pos = torch.tensor(T, dtype=torch.float32, device=positions.device)
                    dirs = positions - cam_pos.unsqueeze(0)
                    dist_xz = torch.norm(dirs[:, [0, 2]], dim=1).clamp(min=1e-6)
                    theta = torch.atan2(dirs[:, 1], dist_xz)
                    phi = torch.atan2(dirs[:, 0], dirs[:, 2])
                    row = ((0.5 - theta / math.pi) * H_g).long().clamp(0, H_g - 1)
                    col = ((0.5 + phi / (2.0 * math.pi)) * W_g).long().clamp(0, W_g - 1)
                    grad_scores_sum += grad_map[row, col]
                    n_grad_frames += 1

            avg_weights = scores_sum / max(len(sampled), 1)

            # ── 5. 最終スコア ──
            scores = avg_weights * gamma * solid_angle * dist_weight

            # ── 6. エラー重み付き補正 exp(-λ * avg_error_norm) ──
            if use_error_weight:
                avg_error = error_weighted_sum / error_weight_total.clamp(min=1e-8)  # [N]
                e_max = torch.quantile(avg_error[avg_error > 0], 0.95).clamp(min=1e-8) if (avg_error > 0).any() else torch.tensor(1.0, device=positions.device)
                error_norm = (avg_error / e_max).clamp(max=1.0)
                accuracy_factor = torch.exp(-error_lambda * error_norm)
                Logger.logInfo(f'  Error weight: avg_error range [{avg_error.min():.4f}, {avg_error.max():.4f}], accuracy_factor range [{accuracy_factor.min():.3f}, {accuracy_factor.max():.3f}]')
                scores = scores * accuracy_factor

            if compute_pixel_grad:
                pixel_grad_scores = grad_scores_sum / max(n_grad_frames, 1) if n_grad_frames > 0 else None
                return scores, pixel_grad_scores
            return scores
    
    def pseudoColorOutputs(self, outputs: dict[str, torch.Tensor | None], camera: 'BaseCamera', dataset: BaseDataset, index: int) -> dict[str, torch.Tensor]:
        """Pseudo-colors the model outputs, returning tensors of shape 3xHxW."""
        return {
            'rgb': outputs['rgb'],
            'depth': pseudoColorDepth(
                color_map='SPECTRAL',
                depth=outputs['depth'],
                near_far=None,
                alpha=outputs['depth'] > 0.0,
                interpolate=True
            ),
        }

    def pseudoColorGT(self, camera: 'BaseCamera', dataset: BaseDataset, index: int) -> dict[str, torch.Tensor]:
        """Pseudo-colors the gt labels relevant for this method, returning tensors of shape 3xHxW."""
        gt_data = {}
        if camera.properties.rgb is not None:
            gt_data['rgb_gt'] = camera.properties.rgb
        return gt_data
