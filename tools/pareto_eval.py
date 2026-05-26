#!/usr/bin/env python3
"""Pareto曲線用：post-hoc pruning 評価スクリプト。

学習済み baseline モデルを読み込み、異なる keep_ratio で
proposed / opacity の2手法でプルーニングし、
(Gaussian数, PSNR) のトレードオフを CSV に保存する。

使用例:
  cd /home/bandailab/nerficg
  python tools/pareto_eval.py --scenes barbershop fisher-hut
"""

import sys
import copy
import argparse
import csv
import re
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

# フレームワークのsrc/をパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import Framework
from Implementations import Methods as MI
from Implementations import Datasets as DI
from Logging import Logger


# ── 設定 ────────────────────────────────────────────────────
KEEP_RATIOS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
N_SAMPLE_FRAMES = 100   # スコア計算に使うフレーム数（速度優先で少なめ）
OUTPUT_CSV = Path('results/pareto_eval.csv')


# ── メトリクス ───────────────────────────────────────────────
def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = F.mse_loss(pred.float(), gt.float()).item()
    return float('inf') if mse < 1e-10 else 10 * np.log10(1.0 / mse)


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    p = pred.float().unsqueeze(0)
    g = gt.float().unsqueeze(0)
    mu1 = F.avg_pool2d(p, 3, 1, 1)
    mu2 = F.avg_pool2d(g, 3, 1, 1)
    s1  = F.avg_pool2d(p * p, 3, 1, 1) - mu1 ** 2
    s2  = F.avg_pool2d(g * g, 3, 1, 1) - mu2 ** 2
    s12 = F.avg_pool2d(p * g, 3, 1, 1) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    val = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / \
          ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
    return val.mean().item()


@torch.no_grad()
def direct_prune(gaussians, keep_mask: torch.Tensor) -> None:
    """optimizerなしで直接テンソルをインデックスしてプルーニング。
    bake_activations() 済みモデル用。
    """
    gaussians._positions = torch.nn.Parameter(gaussians._positions.data[keep_mask])
    gaussians._sh_0      = torch.nn.Parameter(gaussians._sh_0.data[keep_mask])
    gaussians._sh_rest   = torch.nn.Parameter(gaussians._sh_rest.data[keep_mask])
    gaussians._opacities = torch.nn.Parameter(gaussians._opacities.data[keep_mask])
    gaussians._scales    = torch.nn.Parameter(gaussians._scales.data[keep_mask])
    gaussians._rotations = torch.nn.Parameter(gaussians._rotations.data[keep_mask])


@torch.no_grad()
def prune_proposed(gaussians, scores: torch.Tensor, keep_ratio: float,
                   polar_threshold_deg: float = 45.0, far_percentile: float = 0.70) -> None:
    """Region-normalized pruning（optimizerなし版）。"""
    import math
    pos = gaussians._positions.detach()
    n_before = pos.shape[0]

    dist_xz = torch.norm(pos[:, [0, 2]], dim=1).clamp(min=1e-6)
    theta    = torch.atan2(pos[:, 1], dist_xz)
    is_polar = theta.abs() > (polar_threshold_deg * math.pi / 180.0)

    scene_center = pos.mean(dim=0)
    dist    = torch.norm(pos - scene_center, dim=1)
    far_thr = torch.quantile(dist, far_percentile)
    is_far  = dist >= far_thr

    regions = [
        (~is_polar & ~is_far, '近景赤道'),
        ( is_polar & ~is_far, '近景極  '),
        ( is_far,             '遠景    '),
    ]

    normalized_scores = scores.clone().float()
    for mask, _ in regions:
        if mask.sum() < 2:
            continue
        s    = scores[mask].float()
        mean = s.mean()
        std  = s.std().clamp(min=1e-8)
        normalized_scores[mask] = (s - mean) / std

    n_keep = max(1, int(n_before * keep_ratio))
    sorted_indices = torch.argsort(normalized_scores, descending=True)
    keep_mask = torch.zeros(n_before, dtype=torch.bool, device=scores.device)
    keep_mask[sorted_indices[:n_keep]] = True

    direct_prune(gaussians, keep_mask)


@torch.no_grad()
def prune_opacity(gaussians, keep_ratio: float) -> None:
    """Opacity-based pruning（optimizerなし版）。"""
    opacities = gaussians._opacities.detach().squeeze()
    n_keep    = max(1, int(opacities.shape[0] * keep_ratio))
    sorted_indices = torch.argsort(opacities, descending=True)
    keep_mask = torch.zeros(opacities.shape[0], dtype=torch.bool, device=opacities.device)
    keep_mask[sorted_indices[:n_keep]] = True
    direct_prune(gaussians, keep_mask)


@torch.no_grad()
def eval_testset(renderer, dataset) -> dict:
    """テストセット全体のPSNR/SSIMを計算して返す。"""
    dataset.test()
    psnrs, ssims = [], []
    for cam_props in dataset.data[dataset.mode]:
        dataset.camera.setProperties(cam_props)
        out = renderer.renderImage(dataset.camera, to_chw=True)
        pred = out['rgb'].clamp(0, 1)
        gt   = cam_props.rgb.to(pred.device).clamp(0, 1)
        psnrs.append(compute_psnr(pred, gt))
        ssims.append(compute_ssim(pred, gt))
    return {
        'psnr': float(np.mean(psnrs)),
        'ssim': float(np.mean(ssims)),
        'n_gaussians': renderer.model.gaussians.get_positions.shape[0],
    }


def find_baseline_dir(scene: str) -> Path | None:
    """baseline実験ディレクトリ（final.pt あり）を返す。"""
    root = Path('output/SPaGS')
    candidates = sorted(root.glob(f'{scene}_baseline_*'), reverse=True)
    for d in candidates:
        if (d / 'checkpoints' / 'final.pt').exists():
            return d
    return None


def load_model_and_renderer(base_dir: Path):
    """フレームワークをセットアップしてモデル・レンダラー・データセットを返す。"""
    Framework.setup(
        config_path=str(base_dir / 'training_config.yaml'),
        require_custom_config=True,
    )
    dataset = DI.getDataset(
        dataset_type=Framework.config.GLOBAL.DATASET_TYPE,
        path=Framework.config.DATASET.PATH,
    )
    model = MI.getModel(
        method=Framework.config.GLOBAL.METHOD_TYPE,
        checkpoint=str(base_dir / 'checkpoints' / 'final.pt'),
    ).eval()
    renderer = MI.getRenderer(
        method=Framework.config.GLOBAL.METHOD_TYPE,
        model=model,
    )
    return model, renderer, dataset


def main(scenes: list[str]):
    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    rows = []

    for scene in scenes:
        base_dir = find_baseline_dir(scene)
        if base_dir is None:
            print(f'[SKIP] baseline not found for scene: {scene}')
            continue

        print(f'\n{"="*60}')
        print(f'  Scene: {scene}  ({base_dir.name})')
        print(f'{"="*60}')

        # モデル読み込み
        model, renderer, dataset = load_model_and_renderer(base_dir)
        gaussians = model.gaussians

        # 初期テンソルをコピーして保存（state_dictはサイズ変更後にload不可なので直接保存）
        init_tensors = {
            '_positions': gaussians._positions.data.clone(),
            '_sh_0':      gaussians._sh_0.data.clone(),
            '_sh_rest':   gaussians._sh_rest.data.clone(),
            '_opacities': gaussians._opacities.data.clone(),
            '_scales':    gaussians._scales.data.clone(),
            '_rotations': gaussians._rotations.data.clone(),
        }
        n_init = gaussians.get_positions.shape[0]

        # ── no pruning ベースライン点 ──
        print(f'  [baseline] n={n_init:,}', end=' ... ', flush=True)
        metrics = eval_testset(renderer, dataset)
        rows.append({'scene': scene, 'method': 'no_pruning', 'keep_ratio': 1.0, **metrics})
        print(f"PSNR={metrics['psnr']:.3f}")

        # ── contribution score を1回だけ計算（全keep_ratioで共用） ──
        print(f'  Computing contribution scores ({N_SAMPLE_FRAMES} frames)...', flush=True)
        dataset.train()
        scores = renderer.computeSphericalContributionScores(
            dataset=dataset,
            n_sample_frames=N_SAMPLE_FRAMES,
            beta=0.5,
            gamma_min=0.50,
            hf_frame_boost=0.0,
            use_volume=False,
            use_spherical=True,
            use_distance=False,
            distance_lambda=1.0,
            compute_pixel_grad=False,
            use_error_weight=False,
            error_lambda=2.0,
        )

        for keep_ratio in KEEP_RATIOS:
            for method in ['proposed', 'opacity']:
                # 初期テンソルに戻す（Parameterとして再割り当て）
                for attr, tensor in init_tensors.items():
                    setattr(gaussians, attr, torch.nn.Parameter(tensor.clone()))

                # プルーニング適用（optimizer不要の直接実装を使用）
                if method == 'proposed':
                    prune_proposed(gaussians, scores, keep_ratio)
                else:
                    prune_opacity(gaussians, keep_ratio)

                n_after = gaussians.get_positions.shape[0]
                print(f'  [{method}] keep={keep_ratio:.1f} n={n_after:,}', end=' ... ', flush=True)

                metrics = eval_testset(renderer, dataset)
                rows.append({
                    'scene': scene,
                    'method': method,
                    'keep_ratio': keep_ratio,
                    **metrics,
                })
                print(f"PSNR={metrics['psnr']:.3f}")

        # フレームワークのグローバル状態をリセット（次のシーン用）
        Framework._config = None

    # CSV 書き出し
    fieldnames = ['scene', 'method', 'keep_ratio', 'n_gaussians', 'psnr', 'ssim']
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})

    print(f'\nSaved → {OUTPUT_CSV}  ({len(rows)} rows)')

    # サマリー表示
    print(f'\n{"Scene":<25} {"Method":<12} {"keep":<6} {"N_Gauss":>10} {"PSNR":>8}')
    print('-' * 65)
    for r in rows:
        print(f'{r["scene"]:<25} {r["method"]:<12} {r["keep_ratio"]:<6.1f} {r["n_gaussians"]:>10,} {r["psnr"]:>8.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scenes', nargs='+',
        default=['barbershop', 'fisher-hut', 'classroom'],
        help='評価するシーン名（複数可）',
    )
    args = parser.parse_args()
    Logger.setMode(Logger.MODE_SILENT)
    main(args.scenes)
