# -- coding: utf-8 --

"""SPaGS/Trainer.py: Implementation of the trainer for SPaGS."""

import torch
import os
import numpy as np

import Framework
from Datasets.Base import BaseDataset
from Datasets.utils import BasicPointCloud
from Logging import Logger
from Methods.Base.GuiTrainer import GuiTrainer
from Methods.Base.utils import preTrainingCallback, trainingCallback, postTrainingCallback
from Methods.SPaGS.Loss import SPaGSLoss
from Optim.Samplers.DatasetSamplers import DatasetSampler


@Framework.Configurable.configure(
    NUM_ITERATIONS=30_000,
    LEARNING_RATE_POSITION_INIT=0.00016,
    LEARNING_RATE_POSITION_FINAL=0.0000016,
    LEARNING_RATE_POSITION_DELAY_MULT=0.01,
    LEARNING_RATE_POSITION_MAX_STEPS=30_000,
    LEARNING_RATE_FEATURE=0.0025,
    LEARNING_RATE_OPACITY=0.05,
    LEARNING_RATE_SCALING=0.005,
    LEARNING_RATE_ROTATION=0.001,
    PERCENT_DENSE=0.01,
    USE_3D_FILTER=True,
    USE_OPACITY_RESET=True,
    OPACITY_RESET_MAX_OPACITY=0.1,
    USE_OPACITY_DECAY=False,
    USE_VISIBILITY_PRUNING=False,
    VISIBILITY_PRUNING_THRESHOLD=0.01,
    USE_DISTANCE_SCALING=True,
    OPACITY_RESET_INTERVAL=3_000,
    OPACITY_THRESHOLD=0.005,
    DENSIFY_START_ITERATION=200,   # 原著SPaGSは500。200の方がGaussian数が多くなりやすい
    DENSIFY_END_ITERATION=15_000,  # densification終了後にcontribution pruningを行う設計
    DENSIFICATION_INTERVAL=100,
    DENSIFY_GRAD_THRESHOLD=0.00005,  # 原著SPaGSは0.0002。小さいほど densify しやすく Gaussian 数が増える
    USE_WS_LOSS=True,  # True: sin(π·i/H)重み付きL1でWS-PSNRを直接最適化 / False: 均等L1でPSNR最適化
    LOSS=Framework.ConfigParameterList(
        LAMBDA_L1=0.8,
        LAMBDA_DSSIM=0.2,
    ),
    # ── 提案手法：貢献スコアベースプルーニング ──────────────────────────
    USE_CONTRIBUTION_PRUNING=True,
    CONTRIBUTION_PRUNING_KEEP_RATIO=0.632,      # 1ラウンドで残す割合。2ラウンド後の実効保持率 ≈ 0.632^2 ≈ 40%
    CONTRIBUTION_PRUNING_SAMPLE_FRAMES=200,     # スコア計算に使うフレーム数。多いほど安定するが計算コスト増
    CONTRIBUTION_PRUNING_ITERATIONS=[16000, 20000],  # densification終了(15000)直後・学習終了前の2回
    CONTRIBUTION_PRUNING_BETA=0.5,              # 体積項 γ=V_norm^β の指数。0→体積無視、1→線形ペナルティ
    CONTRIBUTION_PRUNING_VOLUME_GAMMA_MIN=0.50, # 体積項の下限。高周波テクスチャを担う小Gaussianの過剰ペナルティを防ぐ
    CONTRIBUTION_PRUNING_HF_FRAME_BOOST=0.0,    # HFフレーム重み付けの強度（0で無効 = 単純フレーム平均）
    USE_HF_DENSIFICATION=False,
    HF_DENSIFICATION_KAPPA=0.3,      # 高周波領域の閾値低減率（0.3 = 70%下げる）
    HF_DENSIFICATION_PERCENTILE=0.8, # 上位20%を高周波領域と判定
    USE_LAT_DENSIFY_CORRECTION=False, # Densification勾配の緯度補正（極の過剰densifyを抑制）
    CONTRIBUTION_PRUNING_USE_VOLUME=True,
    CONTRIBUTION_PRUNING_USE_SPHERICAL=True,        # cos(θ)によるグローバル仰角補正（ワールド座標基準）
    CONTRIBUTION_PRUNING_USE_CAM_SPHERICAL=False,   # cos(θ)をフレームごとのカメラ相対方向で計算（ablation用）
    CONTRIBUTION_PRUNING_USE_DISTANCE=False,        # 遠景ペナルティ。region_normalized で対処するため通常は無効
    CONTRIBUTION_PRUNING_DISTANCE_LAMBDA=1.0,
    USE_PIXEL_GRAD_HF=False,    # 遠景HF判定をGTグラジェント投影で行う（スケールベースより不安定なので無効）
    USE_ERROR_WEIGHT=True,      # 高誤差領域の Gaussian を低スコアにしてプルーニング優先度を上げる
    ERROR_WEIGHT_LAMBDA=2.0,    # exp(-λ·error_norm) の強度。大きいほどエラー領域を積極的に削除
    # ── プルーニング戦略の切り替え（どれか1つ有効にする）───────────────
    USE_LATITUDE_PRUNING=False,       # 提案手法: 緯度帯内 z-score 正規化 → グローバル top-K（極バイアス補正）
    LATITUDE_PRUNING_KEEP_RATIO=0.690,
    LATITUDE_BAND_LOW_DEG=30.0,      # Near-equatorial / Mid-latitude の境界（度）
    LATITUDE_BAND_HIGH_DEG=60.0,     # Mid-latitude / Near-polar の境界（度）
    USE_OPACITY_BASELINE=False,      # ベースライン比較用: opacity 閾値による一様プルーニング
    OPACITY_BASELINE_KEEP_RATIO=0.690,
    USE_OPACITY_SCORE=False,         # ハイブリッド: opacity をスコアとして region-aware に使う（研究用）
    USE_UNIFORM_CONTRIBUTION=False,  # アブレーション: cos補正スコアで region を無視してグローバルtop-K
    CONTRIBUTION_UNIFORM_KEEP_RATIO=0.690,
    USE_REGION_NORMALIZED=True,      # 提案手法: region 内 z-score 正規化 → グローバル top-K（距離バイアス補正）
    REGION_NORMALIZED_KEEP_RATIO=0.690,
    # ── region_aware_pruning のパラメータ（USE_REGION_NORMALIZED=False 時に使用）──
    PRUNING_POLAR_THRESHOLD_DEG=45.0,
    PRUNING_FAR_PERCENTILE=0.70,
    PRUNING_KEEP_NEAR_EQ=0.85,
    PRUNING_KEEP_NEAR_POLAR=0.60,
    PRUNING_KEEP_FAR=0.60,
    PRUNING_HF_SCALE_PERCENTILE=0.50,
    PRUNING_HF_KEEP_BOOST=0.20,
    # ── プルーニング後の position LR リセット ─────────────────────────
    # プルーニングで Gaussian が消えると残った Gaussian の勾配統計が乱れるため、
    # position LR を一時的に大きくしてから減衰させ、再収束を促す。
    PRUNING_LR_RESET_FACTOR=0.30,  # リセット直後の LR = init_lr × factor
    PRUNING_LR_RESET_STEPS=2000,   # この iter 数かけてスケジュール LR に戻す
    # ── SA（Solid-Angle）-Aware opacity pruning ────────────────────────
    # densification 中の opacity プルーニングに仰角補正を加える。
    # 極付近は opacity が高くなりやすいため、cos(θ) で割って threshold を厳しくする。
    SA_OPACITY_PRUNING=False,
    SA_OPACITY_MIN_WEIGHT=0.10,    # cos(θ) の最小クランプ値（ゼロ除算防止）
    # ── 実験管理 ────────────────────────────────────────────────────
    EXPERIMENT_TAG="proposed",     # wandb / output フォルダ名の識別子
)
class SPaGSTrainer(GuiTrainer):
    """Defines the trainer for the SPaGS method."""

    def __init__(self, **kwargs) -> None:
        super(SPaGSTrainer, self).__init__(**kwargs)
        self.train_sampler = None
        self.loss = SPaGSLoss(loss_config=self.LOSS, use_ws_loss=self.USE_WS_LOSS)

    @preTrainingCallback(priority=60)
    def recordTrainingStart(self, *_) -> None:
        """訓練開始時刻を記録し、wandb runにメタ情報を付与する。"""
        import time
        self._training_start_time = time.time()
        if self.WANDB.ACTIVATE:
            import wandb
            wandb.config.update({
                'experiment_tag': self.EXPERIMENT_TAG,
                'n_pruning_iterations': len(self.CONTRIBUTION_PRUNING_ITERATIONS),
                'pruning_keep_ratio': self.CONTRIBUTION_PRUNING_KEEP_RATIO,
                'use_error_weight': self.USE_ERROR_WEIGHT,
                'error_lambda': self.ERROR_WEIGHT_LAMBDA,
                'use_region_normalized': self.USE_REGION_NORMALIZED,
                'use_opacity_baseline': self.USE_OPACITY_BASELINE,
            }, allow_val_change=True)

    @preTrainingCallback(priority=50)
    @torch.no_grad()
    def createSampler(self, _, dataset: 'BaseDataset') -> None: #ランダムにカメラをサンプリングする仕組み
        """Creates the sampler."""
        self.train_sampler = DatasetSampler(dataset=dataset.train(), random=True)

    @preTrainingCallback(priority=40)
    @torch.no_grad()
    def setupGaussians(self, _, dataset: 'BaseDataset') -> None: #初期点群を生成，Datasetに点群があればそれを使う，なければランダムに100k
        """Sets up the model."""
        dataset.train()
        camera_centers = torch.stack([camera_properties.T for camera_properties in dataset])
        radius = (1.1 * torch.max(torch.linalg.norm(camera_centers - torch.mean(camera_centers, dim=0), dim=1))).item()
        # radius = torch.linalg.norm(dataset.point_cloud.positions - torch.mean(camera_centers, dim=0), dim=1).mean().item()
        Logger.logInfo(f'Training cameras extent: {radius:.2f}')

        if dataset.point_cloud is not None:
            point_cloud = dataset.point_cloud
        else:
            n_random_points = 100_000
            min_bounds, max_bounds = dataset.getBoundingBox()
            extent = max_bounds - min_bounds
            point_cloud = BasicPointCloud(torch.rand(n_random_points, 3, dtype=torch.float32, device=min_bounds.device) * extent + min_bounds)
        self.model.gaussians.initialize_from_point_cloud(point_cloud, radius)
        self.model.gaussians.training_setup(self, dataset) #Optimizerなどを初期化
        
    @trainingCallback(priority=110, start_iteration=1000, iteration_stride=1000)
    @torch.no_grad()
    def increaseSHDegree(self, *_) -> None:
        """Increase the number of used SH coefficients up to a maximum degree."""
        self.model.gaussians.increase_used_sh_degree()

    # @trainingCallback(active='USE_VISIBILITY_PRUNING', priority=105, start_iteration=15000, iteration_stride=1000)
    # @torch.no_grad()
    # def importanceBasedPruning(self, iteration: int, dataset: 'BaseDataset') -> None:
    #     """Pruning from RadSplat (see https://arxiv.org/abs/2403.13806)."""
    #     if iteration in [16000, 24000]:
    #         max_blending_weights = self.renderer.computeMaxWeights(dataset.train(), threshold=self.VISIBILITY_PRUNING_THRESHOLD)
    #         self.model.gaussians.importance_pruning(max_blending_weights, threshold=self.VISIBILITY_PRUNING_THRESHOLD)
    @trainingCallback(active='USE_VISIBILITY_PRUNING', priority=105, start_iteration=15000, iteration_stride=1000)
    @torch.no_grad()
    def importanceBasedPruning(self, iteration: int, dataset: 'BaseDataset') -> None:
        if iteration in [16000, 24000]:
            n_before = self.model.gaussians.get_positions.shape[0]
            max_blending_weights = self.renderer.computeMaxWeights(dataset.train(), threshold=self.VISIBILITY_PRUNING_THRESHOLD)
            self.model.gaussians.importance_pruning(max_blending_weights, threshold=self.VISIBILITY_PRUNING_THRESHOLD)
            n_after = self.model.gaussians.get_positions.shape[0]
            self._log_pruning_event(iteration, "opacity_pruning", n_before, n_after)
            
    @trainingCallback(priority=100)
    def trainingIteration(self, iteration: int, dataset: 'BaseDataset') -> None:
        """トレーニングステップ（表示は行わない）"""
        self.model.train()
        dataset.train()
        self.loss.train()
        self.model.gaussians.update_learning_rate(iteration + 1)
        
        camera_properties = self.train_sampler.get(dataset=dataset)['camera_properties']
        dataset.camera.setProperties(camera_properties)
        
        image = self.renderer.renderImageTraining( #本体，トレーニング部分，カメラ1枚サンプル．レンダリング，GTと比較，誤差逆伝搬
            camera=dataset.camera,
            update_densification_info=iteration <= self.DENSIFY_END_ITERATION,
            use_distance_scaling=self.USE_DISTANCE_SCALING,
        )
        
        loss = self.loss(image, camera_properties.rgb)
        loss.backward()
        
        
        # 高周波マップの蓄積（案D用）
        if iteration <= self.DENSIFY_END_ITERATION and self.USE_HF_DENSIFICATION:
            with torch.no_grad():
                # ピクセルごとの誤差マップ [H, W]
                err_map = (image - camera_properties.rgb.to(image.device)).abs().mean(dim=0)
                # 局所分散で高周波度を計算（3×3近傍）
                import torch.nn.functional as F
                err_4d = err_map.unsqueeze(0).unsqueeze(0)          # [1,1,H,W]
                local_mean = F.avg_pool2d(err_4d, 5, stride=1, padding=2)
                local_var  = F.avg_pool2d(err_4d**2, 5, stride=1, padding=2) - local_mean**2
                hf_map = local_var.squeeze().clamp(min=0).sqrt()     # [H, W]
                # グローバルに蓄積（densify()で参照する）
                if not hasattr(self, '_hf_map_accum'):
                    self._hf_map_accum = torch.zeros_like(hf_map)
                    self._hf_map_count = 0
                self._hf_map_accum += hf_map
                self._hf_map_count += 1
        
        if iteration == 14900:
            ops = self.model.gaussians.get_opacities.detach().cpu()
            for thresh in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
                ratio = (ops < thresh).float().mean().item()
                Logger.logInfo(f'  threshold={thresh:.2f}: {ratio*100:.1f}% would be pruned')   
                
                
        # =========================
        # Iterごとの点群保存（PLY版）
        # =========================
        if iteration % 1000 == 0:

            import os

            save_dir = "iter_pointcloud"
            os.makedirs(save_dir, exist_ok=True)

            xyz = self.model.gaussians.get_positions.detach().cpu().numpy()

            scale = None
            if hasattr(self.model.gaussians, "get_scaling"):
                scale = self.model.gaussians.get_scaling.detach().cpu().numpy()

            save_path = os.path.join(save_dir, f"gaussians_{iteration:06d}.ply")

            # -------- PLY書き出し --------
            n = xyz.shape[0]

            with open(save_path, "w") as f:

                # header
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {n}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")

                if scale is not None:
                    f.write("property float scale_0\n")
                    f.write("property float scale_1\n")
                    f.write("property float scale_2\n")

                f.write("end_header\n")

                # body
                if scale is None:
                    for p in xyz:
                        f.write(f"{p[0]} {p[1]} {p[2]}\n")
                else:
                    for p, s in zip(xyz, scale):
                        f.write(f"{p[0]} {p[1]} {p[2]} {s[0]} {s[1]} {s[2]}\n")

            Logger.logInfo(f"[SAVE] Iter {iteration} PLY saved → {save_path}")

        # 最新のロス値を保存（表示はせず、logEarlyConvergenceで使用する）
        self.latest_loss = loss.item() 

    @trainingCallback(priority=5, start_iteration=0, end_iteration=30500, iteration_stride=500)
    @torch.no_grad()
    def logEarlyConvergence(self, iteration: int, dataset: 'BaseDataset') -> None:
        """PSNR, SSIM に加え LPIPS を計算してログ保存"""
        import os
        import torch
        import torch.nn.functional as F
        from torchvision.utils import save_image
        
        self.model.eval()
        test_sample = dataset.test()[0] 
        dataset.camera.setProperties(test_sample)
        device = self.model.gaussians.get_positions.device
        gt_image = test_sample.rgb.to(device) # [3, H, W]
        
        render_out = self.renderer.renderImageTraining(
            camera=dataset.camera,
            update_densification_info=False,
            use_distance_scaling=self.USE_DISTANCE_SCALING,
        ) 
        
        # --- 1. PSNR / SSIM (手動計算) ---
        # mse = F.mse_loss(render_out, gt_image)
        # psnr_val = (20 * torch.log10(1.0 / torch.sqrt(mse))).item() if mse > 0 else 0.0

        def compute_ssim_manual(img1, img2):
            mu1, mu2 = F.avg_pool2d(img1, 3, 1, 1), F.avg_pool2d(img2, 3, 1, 1)
            sigma1_sq = F.avg_pool2d(img1**2, 3, 1, 1) - mu1**2
            sigma2_sq = F.avg_pool2d(img2**2, 3, 1, 1) - mu2**2
            sigma12 = F.avg_pool2d(img1*img2, 3, 1, 1) - mu1*mu2
            C1, C2 = 0.01**2, 0.03**2
            ssim_map = ((2*mu1*mu2 + C1)*(2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1)*(sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean().item()
        
        def compute_ws_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
            """WS-PSNR（球面加重PSNR）の計算。
            
            equirectangularのピクセル重み w(i,j) = sin(π(i+0.5)/H) を使用。
            Sun et al. (2017) IEEE Signal Processing Letters の定義に従う。
            
            Args:
                pred, gt: [3, H, W] float32 tensor, 値域[0,1]
            """
            H, W = pred.shape[-2], pred.shape[-1]
            i = torch.arange(H, device=pred.device, dtype=torch.float32)
            weight = torch.sin(torch.pi * (i + 0.5) / H)  # [H]
            weight = weight / weight.sum() * H             # 正規化
            weight_map = weight[:, None].expand(H, W)      # [H, W]

            mse_map = (pred - gt).pow(2).mean(dim=0)       # [H, W]
            ws_mse = (mse_map * weight_map).sum() / weight_map.sum()
            if ws_mse < 1e-10:
                return 100.0
            return (10 * torch.log10(1.0 / ws_mse)).item()
        
        # --- 1. PSNR / WS-PSNR / SSIM ---
        mse = F.mse_loss(render_out, gt_image)
        psnr_val  = (20 * torch.log10(1.0 / torch.sqrt(mse))).item() if mse > 0 else 0.0
        ws_psnr_val = compute_ws_psnr(render_out, gt_image)  # ← 追加

        ssim_val = compute_ssim_manual(render_out.unsqueeze(0), gt_image.unsqueeze(0))

        # --- 2. LPIPS の計算 ---
        lpips_val = -1.0 # 計算できなかった場合のデフォルト値
        try:
            # torchmetrics を使用する場合
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            # net_type='vgg' または 'alex' (alexの方が高速です)
            # 初回呼び出し時に重みがダウンロードされます
            lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
            # 画像範囲を [0, 1] から [-1, 1] にスケーリングして入力
            lpips_val = lpips_metric(render_out.unsqueeze(0) * 2 - 1, gt_image.unsqueeze(0) * 2 - 1).item()
        except ImportError:
            try:
                # オリジナルの lpips ライブラリを使用する場合
                import lpips
                loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
                lpips_val = loss_fn_vgg(render_out, gt_image).item()
            except ImportError:
                Logger.logInfo(">>> [WARN] LPIPS library not found. Install with 'pip install torchmetrics[image] lpips'")

        # --- 3. ログ保存と表示 ---
        n_points = self.model.gaussians.get_positions.shape[0]
        # --- 1. PSNR / WS-PSNR / SSIM ---
        mse = F.mse_loss(render_out, gt_image)
        psnr_val  = (20 * torch.log10(1.0 / torch.sqrt(mse))).item() if mse > 0 else 0.0
        ws_psnr_val = compute_ws_psnr(render_out, gt_image)  # ← 追加

        # --- 3. ログ保存 ---
        log_file = "quality_metrics_full.csv"
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                f.write("iteration,points,psnr,ws_psnr,ssim,lpips\n")  # ← ws_psnr追加
        with open(log_file, "a") as f:
            f.write(f"{iteration},{n_points},{psnr_val:.4f},{ws_psnr_val:.4f},{ssim_val:.4f},{lpips_val:.4f}\n")

        os.makedirs("render_checks", exist_ok=True)
        save_image(render_out, f"render_checks/iter_{iteration}.png")

        lpips_str = f"{lpips_val:.4f}" if lpips_val >= 0 else "N/A"
        Logger.logInfo(f">>> [EVAL] Iter {iteration:5d} | PSNR: {psnr_val:.2f} | WS-PSNR: {ws_psnr_val:.2f} | SSIM: {ssim_val:.4f} | LPIPS: {lpips_str} | Points: {n_points}")

        if self.WANDB.ACTIVATE:
            import time, wandb
            elapsed = time.time() - getattr(self, '_training_start_time', time.time())
            log_dict = {
                'eval/psnr':       psnr_val,
                'eval/ws_psnr':    ws_psnr_val,
                'eval/ssim':       ssim_val,
                'eval/n_points':   n_points,
                'eval/training_time_min': elapsed / 60.0,
            }
            if lpips_val >= 0:
                log_dict['eval/lpips'] = lpips_val
            wandb.log(log_dict, step=iteration)

        self.model.train()

    # --- 以下、densify や resetOpacities などの既存メソッドが続く ---        
    @trainingCallback(priority=90, start_iteration='DENSIFY_START_ITERATION', end_iteration='DENSIFY_END_ITERATION', iteration_stride='DENSIFICATION_INTERVAL')
    @torch.no_grad()
    def densify(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Apply densification."""
        if iteration == self.DENSIFY_START_ITERATION:
            return

        # 緯度補正：極付近のGaussianに溜まった過剰な2D勾配をcos(θ)でスケールダウン。
        # Equirectangularでは極ほど多くのピクセルが同一方向を向くため勾配が水増しされる。
        if self.USE_LAT_DENSIFY_CORRECTION:
            pos = self.model.gaussians.get_positions.detach()              # [N, 3]
            dist_xz = torch.norm(pos[:, [0, 2]], dim=1).clamp(min=1e-6)   # [N]
            theta = torch.atan2(pos[:, 1], dist_xz)                       # elevation [-π/2, π/2]
            lat_w = torch.cos(theta).clamp(min=0.1)                       # [N] polar min=0.1
            self.model.gaussians.densification_info[1:] *= lat_w[None, :, None]

        self.model.gaussians.densify_and_prune(
            self.DENSIFY_GRAD_THRESHOLD, self.OPACITY_THRESHOLD, prune_large_gaussians=False,
            sa_opacity_pruning=self.SA_OPACITY_PRUNING, sa_min_weight=self.SA_OPACITY_MIN_WEIGHT,
        )

        self.model.gaussians.reduce_opacity(0.001)

        if self.USE_3D_FILTER:
            self.model.gaussians.compute_3d_filter(dataset.train())

    # recoveryDensify は廃止: pruning後に densify すると pruned 数より多く再生成され逆効果。
    # 代わりに contributionBasedPruning で reset_opacities を呼ぶ方式に変更。




    # @trainingCallback(priority=90, start_iteration='DENSIFY_START_ITERATION', end_iteration='DENSIFY_END_ITERATION', iteration_stride='DENSIFICATION_INTERVAL')
    # @torch.no_grad()
    # def densify(self, iteration: int, dataset: 'BaseDataset') -> None:
    #     """高周波適応Densification（案D）。"""
    #     if iteration == self.DENSIFY_START_ITERATION:
    #         return

    #     if self.USE_HF_DENSIFICATION and hasattr(self, '_hf_map_accum') and self._hf_map_count > 0:
    #         # 蓄積した誤差マップを正規化
    #         hf_map = self._hf_map_accum / self._hf_map_count  # [H, W]

    #         # 上位percentileを高周波領域と判定
    #         threshold_val = torch.quantile(hf_map, self.HF_DENSIFICATION_PERCENTILE)
    #         is_hf_pixel = hf_map > threshold_val               # [H, W]  bool

    #         # Gaussianを現在のカメラでスクリーン空間に投影して高周波ピクセルと対応付け
    #         # → 簡易版：densification_infoの勾配をGaussianごとにスケール
    #         hf_scale = torch.ones(
    #             self.model.gaussians.get_positions.shape[0],
    #             device='cuda', dtype=torch.float32
    #         )

    #         # 各Gaussianの2D投影位置から高周波フラグを引く
    #         # SPaGSのdensification_infoはカウント[0]と勾配[1],[2]を持つ
    #         # 位置勾配が大きいGaussianが多いピクセルほどhf_mapが大きいはずなので
    #         # 勾配そのものにスケールをかける
    #         denominator = self.model.gaussians.densification_info[0].clamp_min(1.0)
    #         grads = self.model.gaussians.densification_info[1] / denominator  # [N, 1]

    #         # hf_mapのグローバル統計でGaussianの勾配閾値を調整
    #         hf_ratio = is_hf_pixel.float().mean().item()  # 高周波ピクセルの割合

    #         # 高周波領域が多いフレームほど全体の閾値を下げる（密化促進）
    #         adaptive_threshold = self.DENSIFY_GRAD_THRESHOLD * (
    #             1.0 - (1.0 - self.HF_DENSIFICATION_KAPPA) * hf_ratio
    #         )

    #         Logger.logInfo(f'  [HF-Densify] iter={iteration}, hf_ratio={hf_ratio:.3f}, grad_thresh={adaptive_threshold:.6f}')

    #         self.model.gaussians.densify_and_prune(
    #             adaptive_threshold,
    #             self.OPACITY_THRESHOLD,
    #             prune_large_gaussians=False
    #         )

    #         # バッファリセット
    #         del self._hf_map_accum
    #         del self._hf_map_count

    #     else:
    #         # USE_HF_DENSIFICATION=False のときは元の動作
    #         self.model.gaussians.densify_and_prune(
    #             self.DENSIFY_GRAD_THRESHOLD,
    #             self.OPACITY_THRESHOLD,
    #             prune_large_gaussians=False
    #         )

    #     self.model.gaussians.reduce_opacity(0.001)

    #     if self.USE_3D_FILTER:
    #         self.model.gaussians.compute_3d_filter(dataset.train())
            
            
            

    @trainingCallback(active='USE_OPACITY_RESET', priority=80, start_iteration='OPACITY_RESET_INTERVAL', end_iteration='DENSIFY_END_ITERATION', iteration_stride='OPACITY_RESET_INTERVAL')
    @torch.no_grad()
    def resetOpacities(self, iteration: int, _) -> None:
        """Reset opacities."""
        if iteration == self.DENSIFY_END_ITERATION:
            return
        self.model.gaussians.reset_opacities(max_opacity=self.OPACITY_RESET_MAX_OPACITY)

    @trainingCallback(active='USE_OPACITY_DECAY', priority=80, start_iteration='DENSIFY_START_ITERATION', end_iteration='DENSIFY_END_ITERATION', iteration_stride=50)
    @torch.no_grad()
    def decayOpacities(self, iteration: int, _) -> None:
        """Decay opacities."""
        if iteration == self.DENSIFY_START_ITERATION:
            return
        self.model.gaussians.decay_opacities(decay_factor=0.9995)

    @trainingCallback(active='USE_3D_FILTER', priority=75, start_iteration='DENSIFY_END_ITERATION', iteration_stride=100)
    @torch.no_grad()
    def recompute3DFilter(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Recompute 3D filter."""
        if self.DENSIFY_END_ITERATION < iteration < self.NUM_ITERATIONS - 100:
            self.model.gaussians.compute_3d_filter(dataset.train())

    @trainingCallback(priority=70)
    @torch.no_grad()
    def performOptimizerStep(self, *_) -> None:
        """Update parameters."""
        self.model.gaussians.optimizer.step()
        self.model.gaussians.optimizer.zero_grad()

    @trainingCallback(active='WANDB.ACTIVATE', priority=10, iteration_stride='WANDB.INTERVAL')
    @torch.no_grad()
    def logWandB(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Adds primitive count to default Weights & Biases logging."""
        Framework.wandb.log({
            'n_primitives': self.model.gaussians.get_positions.shape[0]
        }, step=iteration)
        # default logging
        super().logWandB(iteration, dataset)

    @trainingCallback(active='WANDB.ACTIVATE', priority=1, iteration_stride='WANDB.INTERVAL')
    def commitWandB(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Base の commitWandB をオーバーライドし、explicit step を渡す。

        Base の実装は step を省略するため、auto-step タイムラインと explicit-step
        タイムラインが混在し、iter 15000 以降のデータが WandB チャートに表示されない
        問題が発生する。step=iteration を明示することで両者を統一する。
        """
        Framework.wandb.log(data={}, commit=True, step=iteration)

    @postTrainingCallback(priority=1000)
    @torch.no_grad()
    def bakeActivations(self, *_) -> None:
        """Bake relevant activation functions after training."""
        self.model.gaussians.bake_activations()
        self.model.gaussians.optimizer = None

    @postTrainingCallback(priority=999)
    @torch.no_grad()
    def logFinalStats(self, *_) -> None:
        """訓練終了時の最終統計をwandbに記録する。"""
        if not self.WANDB.ACTIVATE:
            return
        import time, wandb
        elapsed = time.time() - getattr(self, '_training_start_time', time.time())
        n_final = self.model.gaussians.get_positions.shape[0]

        # モデルサイズ（Gaussianパラメータのメモリ量）
        size_bytes = sum(p.numel() * p.element_size() for p in self.model.gaussians.parameters())
        size_mb = size_bytes / 1024 / 1024

        wandb.log({
            'final/n_gaussians':        n_final,
            'final/model_size_mb':      size_mb,
            'final/training_time_min':  elapsed / 60.0,
            'final/training_time_sec':  elapsed,
        })
        wandb.run.summary['final_n_gaussians']       = n_final
        wandb.run.summary['final_model_size_mb']     = size_mb
        wandb.run.summary['total_training_time_min'] = elapsed / 60.0
        wandb.run.summary['experiment_tag']          = self.EXPERIMENT_TAG
        Logger.logInfo(f'[wandb] Final stats logged: {n_final:,} Gaussians, {size_mb:.1f} MB, {elapsed/60:.1f} min')
    
    #ここ追加
    @trainingCallback(
        active='USE_CONTRIBUTION_PRUNING',
        priority=95,
        start_iteration=9000,   # densification中（10000/13000）に発火できるよう早める
        iteration_stride=1000,
    )
    @torch.no_grad()
    def contributionBasedPruning(self, iteration: int, dataset: 'BaseDataset') -> None:
        """球面 Contribution-based Pruning（立体角補正あり）。

        タイミングの設計意図：
          iter 16000: densification 終了(15000)直後。Gaussian 数がピークに達した状態で
                      1回目のプルーニングを行い、不要 Gaussian を大量に除去する。
          iter 20000: 1回目プルーニング後に再収束した状態で2回目を行い、
                      精度を維持しながらさらに圧縮する。
          iter 30000: 学習終了（プルーニングなし）。
        """
        if iteration not in self.CONTRIBUTION_PRUNING_ITERATIONS:
            return

        n_before = self.model.gaussians.get_positions.shape[0]

        # 最初のプルーニング直前の位置を保存（緯度分布の before/after 比較用）
        if iteration == self.CONTRIBUTION_PRUNING_ITERATIONS[0]:
            import numpy as np
            pos_np = self.model.gaussians.get_positions.detach().cpu().numpy()
            save_path = str(self.output_directory / f'positions_pre_pruning_{iteration}.npy')
            np.save(save_path, pos_np)
            Logger.logInfo(f'  [SAVE] Pre-pruning positions → {save_path} (N={len(pos_np):,})')

        # ── 提案手法: 緯度帯内 z-score 正規化 → グローバル top-K ──
        if self.USE_LATITUDE_PRUNING:
            Logger.logInfo(f'[iter {iteration}] Latitude-band pruning (keep={self.LATITUDE_PRUNING_KEEP_RATIO:.3f})...')
            self.model.gaussians.latitude_band_pruning(
                keep_ratio=self.LATITUDE_PRUNING_KEEP_RATIO,
                lat_band_low=self.LATITUDE_BAND_LOW_DEG,
                lat_band_high=self.LATITUDE_BAND_HIGH_DEG,
            )
            n_after = self.model.gaussians.get_positions.shape[0]
            self._log_pruning_event(iteration, "latitude_band_pruning", n_before, n_after)
            if self.USE_3D_FILTER:
                self.model.gaussians.compute_3d_filter(dataset.train())
            self.model.gaussians.reset_opacities(max_opacity=self.OPACITY_RESET_MAX_OPACITY)
            Logger.logInfo(f'  Post-pruning opacity reset (max={self.OPACITY_RESET_MAX_OPACITY})')
            if iteration <= self.DENSIFY_END_ITERATION:
                self.model.gaussians.reset_densification_info()
            reset_lr = (self.LEARNING_RATE_POSITION_INIT
                        * self.model.gaussians.training_cameras_extent
                        * self.PRUNING_LR_RESET_FACTOR)
            self.model.gaussians.set_post_pruning_lr(reset_lr, iteration, self.PRUNING_LR_RESET_STEPS)
            return

        # ── ベースライン: opacityベース一様プルーニング ──
        if self.USE_OPACITY_BASELINE:
            Logger.logInfo(f'[iter {iteration}] Opacity-baseline pruning (keep={self.OPACITY_BASELINE_KEEP_RATIO:.3f})...')
            self.model.gaussians.opacity_pruning(keep_ratio=self.OPACITY_BASELINE_KEEP_RATIO)
            n_after = self.model.gaussians.get_positions.shape[0]
            self._log_pruning_event(iteration, "opacity_baseline_pruning", n_before, n_after)
            if self.USE_3D_FILTER:
                self.model.gaussians.compute_3d_filter(dataset.train())
            self.model.gaussians.reset_opacities(max_opacity=self.OPACITY_RESET_MAX_OPACITY)
            Logger.logInfo(f'  Post-pruning opacity reset (max={self.OPACITY_RESET_MAX_OPACITY})')
            if iteration <= self.DENSIFY_END_ITERATION:
                self.model.gaussians.reset_densification_info()
            reset_lr = (self.LEARNING_RATE_POSITION_INIT
                        * self.model.gaussians.training_cameras_extent
                        * self.PRUNING_LR_RESET_FACTOR)
            self.model.gaussians.set_post_pruning_lr(reset_lr, iteration, self.PRUNING_LR_RESET_STEPS)
            return

        # ── ハイブリッド: opacityをスコアとしてregion-aware pruningに使用 ──
        if self.USE_OPACITY_SCORE:
            Logger.logInfo(f'[iter {iteration}] Using opacity as region-aware pruning score...')
            scores = self.model.gaussians.get_opacities.squeeze().detach()
            pixel_grad_scores = None
            self.model.gaussians.region_aware_pruning(
                scores=scores,
                polar_threshold_deg=self.PRUNING_POLAR_THRESHOLD_DEG,
                keep_near_eq=self.PRUNING_KEEP_NEAR_EQ,
                keep_near_polar=self.PRUNING_KEEP_NEAR_POLAR,
                keep_far=self.PRUNING_KEEP_FAR,
                far_percentile=self.PRUNING_FAR_PERCENTILE,
                hf_scale_percentile=self.PRUNING_HF_SCALE_PERCENTILE,
                hf_keep_boost=self.PRUNING_HF_KEEP_BOOST,
                target_keep_ratio=self.CONTRIBUTION_PRUNING_KEEP_RATIO,
                pixel_grad_scores=pixel_grad_scores,
            )
            n_after = self.model.gaussians.get_positions.shape[0]
            self._log_pruning_event(iteration, "opacity_region_pruning", n_before, n_after)
            if self.USE_3D_FILTER:
                self.model.gaussians.compute_3d_filter(dataset.train())
            self.model.gaussians.reset_opacities(max_opacity=self.OPACITY_RESET_MAX_OPACITY)
            Logger.logInfo(f'  Post-pruning opacity reset (max={self.OPACITY_RESET_MAX_OPACITY})')
            if iteration <= self.DENSIFY_END_ITERATION:
                self.model.gaussians.reset_densification_info()
            reset_lr = (self.LEARNING_RATE_POSITION_INIT
                        * self.model.gaussians.training_cameras_extent
                        * self.PRUNING_LR_RESET_FACTOR)
            self.model.gaussians.set_post_pruning_lr(reset_lr, iteration, self.PRUNING_LR_RESET_STEPS)
            return

        Logger.logInfo(f'[iter {iteration}] Computing spherical contribution scores ({self.CONTRIBUTION_PRUNING_SAMPLE_FRAMES} frames)...')

        score_result = self.renderer.computeSphericalContributionScores(
            dataset=dataset.train(),
            n_sample_frames=self.CONTRIBUTION_PRUNING_SAMPLE_FRAMES,
            beta=self.CONTRIBUTION_PRUNING_BETA,
            gamma_min=self.CONTRIBUTION_PRUNING_VOLUME_GAMMA_MIN,
            hf_frame_boost=self.CONTRIBUTION_PRUNING_HF_FRAME_BOOST,
            use_volume=self.CONTRIBUTION_PRUNING_USE_VOLUME,
            use_spherical=self.CONTRIBUTION_PRUNING_USE_SPHERICAL,
            use_cam_spherical=self.CONTRIBUTION_PRUNING_USE_CAM_SPHERICAL,
            use_distance=self.CONTRIBUTION_PRUNING_USE_DISTANCE,
            distance_lambda=self.CONTRIBUTION_PRUNING_DISTANCE_LAMBDA,
            compute_pixel_grad=self.USE_PIXEL_GRAD_HF,
            use_error_weight=self.USE_ERROR_WEIGHT,
            error_lambda=self.ERROR_WEIGHT_LAMBDA,
        )
        if self.USE_PIXEL_GRAD_HF:
            scores, pixel_grad_scores = score_result
        else:
            scores, pixel_grad_scores = score_result, None

        # ── アブレーション: contribution scoreで一様グローバルプルーニング ──
        if self.USE_UNIFORM_CONTRIBUTION:
            Logger.logInfo(f'[iter {iteration}] Uniform contribution pruning (keep={self.CONTRIBUTION_UNIFORM_KEEP_RATIO:.3f})...')
            self.model.gaussians.uniform_score_pruning(scores, keep_ratio=self.CONTRIBUTION_UNIFORM_KEEP_RATIO)
            n_after = self.model.gaussians.get_positions.shape[0]
            self._log_pruning_event(iteration, "uniform_contribution_pruning", n_before, n_after)
            if self.USE_3D_FILTER:
                self.model.gaussians.compute_3d_filter(dataset.train())
            self.model.gaussians.reset_opacities(max_opacity=self.OPACITY_RESET_MAX_OPACITY)
            Logger.logInfo(f'  Post-pruning opacity reset (max={self.OPACITY_RESET_MAX_OPACITY})')
            if iteration <= self.DENSIFY_END_ITERATION:
                self.model.gaussians.reset_densification_info()
            reset_lr = (self.LEARNING_RATE_POSITION_INIT
                        * self.model.gaussians.training_cameras_extent
                        * self.PRUNING_LR_RESET_FACTOR)
            self.model.gaussians.set_post_pruning_lr(reset_lr, iteration, self.PRUNING_LR_RESET_STEPS)
            return

        # ── 提案手法: region内z-score正規化→グローバルtop-K（距離biasを補正）──
        if self.USE_REGION_NORMALIZED:
            Logger.logInfo(f'[iter {iteration}] Region-normalized contribution pruning (keep={self.REGION_NORMALIZED_KEEP_RATIO:.3f})...')
            self.model.gaussians.region_normalized_pruning(
                scores=scores,
                keep_ratio=self.REGION_NORMALIZED_KEEP_RATIO,
                polar_threshold_deg=self.PRUNING_POLAR_THRESHOLD_DEG,
                far_percentile=self.PRUNING_FAR_PERCENTILE,
            )
            n_after = self.model.gaussians.get_positions.shape[0]
            self._log_pruning_event(iteration, "region_normalized_pruning", n_before, n_after)
            if self.USE_3D_FILTER:
                self.model.gaussians.compute_3d_filter(dataset.train())
            self.model.gaussians.reset_opacities(max_opacity=self.OPACITY_RESET_MAX_OPACITY)
            Logger.logInfo(f'  Post-pruning opacity reset (max={self.OPACITY_RESET_MAX_OPACITY})')
            if iteration <= self.DENSIFY_END_ITERATION:
                self.model.gaussians.reset_densification_info()
            reset_lr = (self.LEARNING_RATE_POSITION_INIT
                        * self.model.gaussians.training_cameras_extent
                        * self.PRUNING_LR_RESET_FACTOR)
            self.model.gaussians.set_post_pruning_lr(reset_lr, iteration, self.PRUNING_LR_RESET_STEPS)
            return

        self.model.gaussians.region_aware_pruning(
            scores=scores,
            polar_threshold_deg=self.PRUNING_POLAR_THRESHOLD_DEG,
            keep_near_eq=self.PRUNING_KEEP_NEAR_EQ,
            keep_near_polar=self.PRUNING_KEEP_NEAR_POLAR,
            keep_far=self.PRUNING_KEEP_FAR,
            far_percentile=self.PRUNING_FAR_PERCENTILE,
            hf_scale_percentile=self.PRUNING_HF_SCALE_PERCENTILE,
            hf_keep_boost=self.PRUNING_HF_KEEP_BOOST,
            target_keep_ratio=self.CONTRIBUTION_PRUNING_KEEP_RATIO,
            pixel_grad_scores=pixel_grad_scores,
        )
        n_after = self.model.gaussians.get_positions.shape[0]
        self._log_pruning_event(iteration, "region_aware_pruning", n_before, n_after)

        if self.USE_3D_FILTER:
            self.model.gaussians.compute_3d_filter(dataset.train())

        # Opacity reset: 残存Gaussianを解放して再適応を促す
        self.model.gaussians.reset_opacities(max_opacity=self.OPACITY_RESET_MAX_OPACITY)
        Logger.logInfo(f'  Post-pruning opacity reset (max={self.OPACITY_RESET_MAX_OPACITY})')

        # densification中にpruningした場合: densification_infoのサイズを新N点に合わせる
        # これがないと次のdensify stepでshape mismatchエラーになる
        if iteration <= self.DENSIFY_END_ITERATION:
            self.model.gaussians.reset_densification_info()
            Logger.logInfo(f'  densification_info reset for continued densification')

        # Position LRリセット: pruning後に形状的な回復を可能にする
        # iter 10000/13000 時点でLRは既にかなり減衰しているため、形状が動けない
        reset_lr = (self.LEARNING_RATE_POSITION_INIT
                    * self.model.gaussians.training_cameras_extent
                    * self.PRUNING_LR_RESET_FACTOR)
        self.model.gaussians.set_post_pruning_lr(reset_lr, iteration, self.PRUNING_LR_RESET_STEPS)
    
    def _log_pruning_event(self, iteration: int, method: str, n_before: int, n_after: int) -> None:
        """PruningイベントをCSVおよびwandbに記録する。"""
        import os
        log_file = "pruning_log.csv"
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                f.write("iteration,method,n_before,n_after,reduction_pct\n")
        reduction = (n_before - n_after) / n_before * 100
        with open(log_file, "a") as f:
            f.write(f"{iteration},{method},{n_before},{n_after},{reduction:.2f}\n")
        Logger.logInfo(f'  [{method}] {n_before:,} → {n_after:,} ({reduction:.1f}% removed)')
        if self.WANDB.ACTIVATE:
            import wandb
            wandb.log({
                'pruning/n_before':      n_before,
                'pruning/n_after':       n_after,
                'pruning/reduction_pct': reduction,
            }, step=iteration)
