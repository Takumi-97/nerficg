#!/usr/bin/env python3
"""仰角バイアスの実証実験：スコアと仰角の相関を可視化。

equirectangularでは極付近に1/cos(θ)倍のピクセルが集中するため、
cos補正なしのGaussian貢献スコアが極ほど過大評価される。
このスクリプトはその実態を実験的に示す。

使用例:
  python tools/plot_score_bias.py
  python tools/plot_score_bias.py --scene barbershop --method proposed_camcos
"""

import argparse
import sys
import os
import random
import math
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_ROOT = Path('output/SPaGS')


def find_exp_dir(scene: str, method: str) -> Path | None:
    candidates = sorted(OUTPUT_ROOT.glob(f'{scene}_{method}_*'), reverse=True)
    for d in candidates:
        if (d / 'checkpoints' / 'final.pt').exists():
            return d
    return None


def compute_score_vs_elevation(exp_dir: Path, n_frames: int = 100, seed: int = 42):
    """
    フレームをサンプリングしてGaussianのブレンド重みを集計し、
    cos補正あり/なしのスコアと仰角の関係を返す。
    """
    import torch
    import yaml

    # フレームワーク初期化
    sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    import utils
    with utils.discoverSourcePath():
        import Framework
        from Implementations import Methods as MI, Datasets as DI

    config_path = str(exp_dir / 'training_config.yaml')
    Framework.setup(config_path=config_path, require_custom_config=True)

    # Trainerをチェックポイントなしで初期化してからstate dictを手動ロード
    model_instance = MI.getTrainingInstance(
        method=Framework.config.GLOBAL.METHOD_TYPE,
        checkpoint=None,
    )
    pt_path = exp_dir / 'checkpoints' / 'final.pt'
    ckpt = torch.load(str(pt_path), map_location='cuda', weights_only=False)
    state = ckpt['model_state_dict']
    # register_parameter('_positions', None) で初期化されるためload_state_dictが機能しない。
    # nn.Parameterとして直接代入する。
    g = model_instance.model.gaussians
    g._positions  = torch.nn.Parameter(state['gaussians._positions'].cuda())
    g._scales     = torch.nn.Parameter(state['gaussians._scales'].cuda())
    g._rotations  = torch.nn.Parameter(state['gaussians._rotations'].cuda())
    g._opacities  = torch.nn.Parameter(state['gaussians._opacities'].cuda())
    if 'gaussians._sh_0' in state:
        g._sh_0   = torch.nn.Parameter(state['gaussians._sh_0'].cuda())
    if 'gaussians._sh_rest' in state:
        g._sh_rest = torch.nn.Parameter(state['gaussians._sh_rest'].cuda())
    print(f'  Loaded gaussians: N={g._positions.shape[0]:,} from {pt_path.name}')
    dataset = DI.getDataset(
        dataset_type=Framework.config.GLOBAL.DATASET_TYPE,
        path=Framework.config.DATASET.PATH,
    )
    dataset.setMode('train')

    renderer  = model_instance.renderer
    gaussians = model_instance.model.gaussians

    positions  = gaussians.get_positions.detach()          # [N, 3]
    scales     = gaussians.get_scales.detach()
    rotations  = gaussians.get_rotations.detach()
    opacities  = gaussians.get_opacities.detach()
    n          = positions.shape[0]

    # 仰角 θ（ワールド座標）
    dist_xz = torch.norm(positions[:, [0, 2]], dim=1).clamp(min=1e-6)
    theta   = torch.atan2(positions[:, 1], dist_xz)        # [-π/2, π/2]
    elev_deg = (theta * 180.0 / math.pi).cpu().numpy()      # [N]
    cos_theta = torch.cos(theta).clamp(min=0.1).cpu().numpy()

    # フレームをサンプリングしてブレンド重みを累積
    random.seed(seed)
    all_props = list(dataset.data[dataset.mode])
    sampled   = random.sample(all_props, min(n_frames, len(all_props)))

    scores_sum = torch.zeros(n, device=positions.device)
    for props in sampled:
        dataset.camera.setProperties(props)
        fw = torch.zeros(n, device=positions.device)
        renderer.rasterizer.update_max_weights(
            max_weights=fw,
            positions=positions,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            camera=dataset.camera,
            mode=renderer.BLEND_MODE,
            K=renderer.K,
            active_sh_bases=0,
            scale_modifier=1.0,
            weight_threshold=0.01,
        )
        scores_sum += fw

    Framework.teardown()

    avg_weight  = (scores_sum / len(sampled)).cpu().numpy()   # 補正なしスコア
    corrected   = avg_weight * cos_theta                       # cos補正後スコア

    return elev_deg, avg_weight, corrected, cos_theta


def plot_bias(elev_deg, raw_score, corrected, output_path: Path, scene: str):
    """散布図＋ビン平均＋理論曲線でバイアスを可視化。"""
    bins     = np.linspace(-90, 90, 37)
    centers  = (bins[:-1] + bins[1:]) / 2
    cos_theory = np.cos(np.radians(centers)).clip(min=0.01)

    # ビン平均
    raw_binned  = np.zeros(len(centers))
    corr_binned = np.zeros(len(centers))
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (elev_deg >= lo) & (elev_deg < hi) & (raw_score > 0)
        if mask.sum() > 0:
            raw_binned[i]  = raw_score[mask].mean()
            corr_binned[i] = corrected[mask].mean()

    # 正規化（赤道基準）
    eq_mask = np.abs(centers) < 15
    raw_norm  = raw_binned  / raw_binned[eq_mask].mean().clip(1e-9)
    corr_norm = corr_binned / corr_binned[eq_mask].mean().clip(1e-9)
    theory_norm = (1 / cos_theory) / (1 / cos_theory[eq_mask]).mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── 左：補正なしスコア vs 仰角（バイアスの実証）──
    ax = axes[0]
    # 散布図（間引き）
    idx = np.where(raw_score > 0)[0]
    if len(idx) > 20000:
        idx = np.random.choice(idx, 20000, replace=False)
    ax.scatter(elev_deg[idx], raw_score[idx], s=1, alpha=0.15, color='#888888', rasterized=True)
    ax.plot(centers, raw_binned, color='#E05A2B', linewidth=2.5, label='Binned mean (raw score)')
    ax2 = ax.twinx()
    ax2.plot(centers, 1 / cos_theory, color='#4477AA', linewidth=2, linestyle='--',
             label='1/cos(θ) — theory')
    ax2.set_ylabel('1/cos(θ)', fontsize=10, color='#4477AA')
    ax2.tick_params(axis='y', colors='#4477AA')
    ax.set_xlabel('Elevation Angle (degrees)', fontsize=11)
    ax.set_ylabel('Raw Contribution Score (α·T avg)', fontsize=10)
    ax.set_xlim(-90, 90)
    ax.set_xticks(range(-90, 91, 30))
    ax.axvspan(-90, -60, alpha=0.06, color='blue')
    ax.axvspan(60,   90, alpha=0.06, color='blue')
    ax.set_title(f'{scene} — Raw Score vs Elevation\n(polar bias visible)', fontsize=12)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper center')
    ax.grid(linestyle='--', alpha=0.4)

    # ── 右：補正前後の正規化スコア比較──
    ax = axes[1]
    ax.plot(centers, raw_norm,   color='#E05A2B', linewidth=2.5, label='Without correction')
    ax.plot(centers, corr_norm,  color='#2CA02C', linewidth=2.5, label='With cos(θ) correction')
    ax.plot(centers, theory_norm, color='#4477AA', linewidth=1.5, linestyle='--',
            label='1/cos(θ) theory')
    ax.axhline(1.0, color='#AAAAAA', linestyle=':', linewidth=1, label='Uniform reference')
    ax.axvspan(-90, -60, alpha=0.06, color='blue')
    ax.axvspan(60,   90, alpha=0.06, color='blue')
    ax.set_xlabel('Elevation Angle (degrees)', fontsize=11)
    ax.set_ylabel('Normalized Score (equator = 1.0)', fontsize=10)
    ax.set_xlim(-90, 90)
    ax.set_xticks(range(-90, 91, 30))
    ax.set_title(f'{scene} — Normalized Score: Before vs After Correction', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(linestyle='--', alpha=0.4)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {output_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene',  default='barbershop')
    parser.add_argument('--method', default='proposed_camcos')
    parser.add_argument('--n_frames', type=int, default=100)
    parser.add_argument('--outdir',   type=Path, default=Path('results/figures'))
    args = parser.parse_args()

    exp_dir = find_exp_dir(args.scene, args.method)
    if exp_dir is None:
        print(f'ERROR: {args.scene}/{args.method} not found or no final.pt')
        return

    print(f'Using: {exp_dir}')
    elev, raw, corr, cos_t = compute_score_vs_elevation(exp_dir, n_frames=args.n_frames)

    out = args.outdir / f'score_bias_{args.scene}.png'
    plot_bias(elev, raw, corr, out, args.scene)


if __name__ == '__main__':
    main()
