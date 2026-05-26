#!/usr/bin/env python3
"""Gaussianの緯度分布ヒストグラムを生成。

final.pt チェックポイントから Gaussian の3D位置を読み込み、
緯度（elevation angle）分布をヒストグラムで可視化する。
positions_pre_pruning_*.npy が存在する場合はプルーニング前後の比較も描画する。

使用例:
  python tools/plot_gauss_dist.py
  python tools/plot_gauss_dist.py --scenes barbershop lone_monk
  python tools/plot_gauss_dist.py --before_after  # 同一実験でbefore/after比較
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_ROOT = Path('output/SPaGS')

COMPARE_METHODS = [
    ('spags',       'SPaGS',     '#888888'),
    ('baseline',    'Baseline',  '#4477AA'),
    ('proposed_v1', 'Proposed',  '#E05A2B'),
]


def find_exp_dir(scene: str, method_tag: str) -> Path | None:
    candidates = sorted(OUTPUT_ROOT.glob(f'{scene}_{method_tag}_*'), reverse=True)
    for d in candidates:
        if (d / 'checkpoints' / 'final.pt').exists():
            return d
    return None


def load_positions_from_pt(pt_path: Path) -> np.ndarray | None:
    try:
        import torch
        ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        for key, val in state.items():
            if 'position' in key.lower() and hasattr(val, 'shape') and len(val.shape) == 2:
                return val.numpy() if hasattr(val, 'numpy') else val
        return None
    except Exception as e:
        print(f'  Warning: could not load {pt_path}: {e}')
        return None


def load_positions_pre(exp_dir: Path) -> np.ndarray | None:
    """positions_pre_pruning_*.npy を探して読み込む。"""
    candidates = sorted(exp_dir.glob('positions_pre_pruning_*.npy'))
    if not candidates:
        return None
    return np.load(candidates[0])


def positions_to_elevation(pos: np.ndarray) -> np.ndarray:
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    dist_xz = np.sqrt(x**2 + z**2).clip(min=1e-6)
    return np.degrees(np.arctan2(y, dist_xz))


def elevation_to_density(elev: np.ndarray, bins: int = 36) -> tuple:
    counts, edges = np.histogram(elev, bins=bins, range=(-90, 90))
    centers = (edges[:-1] + edges[1:]) / 2
    solid_angle = np.cos(np.radians(centers)).clip(min=1e-6)
    density = counts / (solid_angle * counts.sum())
    density /= density.mean()
    return centers, density


def add_band_shading(ax):
    ax.axvspan(-90, -60, alpha=0.06, color='blue')
    ax.axvspan(60,   90, alpha=0.06, color='blue')
    ax.axvspan(-30,  30, alpha=0.06, color='green')
    ax.axhline(1.0, color='#AAAAAA', linestyle='--', linewidth=1, label='Uniform (reference)')


# ── 手法間比較（従来モード）────────────────────────────────────────
def plot_method_comparison(scene: str, methods: list, output_dir: Path, bins: int = 36):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    add_band_shading(ax)

    any_plotted = False
    for tag, label, color in methods:
        d = find_exp_dir(scene, tag)
        if d is None:
            continue
        pos = load_positions_from_pt(d / 'checkpoints' / 'final.pt')
        if pos is None:
            continue
        elev = positions_to_elevation(pos)
        centers, density = elevation_to_density(elev, bins)
        ax.plot(centers, density, color=color, linewidth=1.8,
                label=f'{label} (N={len(elev)/1e3:.0f}K)')
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    ax.set_xlabel('Elevation Angle (degrees)', fontsize=12)
    ax.set_ylabel('Normalized Density\n(relative to uniform sphere)', fontsize=10)
    ax.set_xlim(-90, 90)
    ax.set_xticks(range(-90, 91, 15))
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_title(f'{scene} — Gaussian Elevation Distribution (post-pruning)', fontsize=13)

    out = output_dir / f'gauss_dist_{scene}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out}')


# ── Before/After 比較（同一実験内）───────────────────────────────
def plot_before_after(scene: str, method_tag: str, label: str, color: str,
                      output_dir: Path, bins: int = 36):
    d = find_exp_dir(scene, method_tag)
    if d is None:
        print(f'  SKIP: {scene}/{method_tag} not found')
        return

    pos_after = load_positions_from_pt(d / 'checkpoints' / 'final.pt')
    pos_before = load_positions_pre(d)

    if pos_after is None:
        print(f'  SKIP: final.pt not loadable for {scene}/{method_tag}')
        return
    if pos_before is None:
        print(f'  INFO: no pre_pruning.npy for {scene}/{method_tag} — skipping before/after plot')
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    add_band_shading(ax)

    elev_before = positions_to_elevation(pos_before)
    elev_after  = positions_to_elevation(pos_after)

    c_b, d_b = elevation_to_density(elev_before, bins)
    c_a, d_a = elevation_to_density(elev_after, bins)

    ax.plot(c_b, d_b, color='#AAAAAA', linewidth=1.8, linestyle='--',
            label=f'Before pruning (N={len(elev_before)/1e3:.0f}K)')
    ax.plot(c_a, d_a, color=color, linewidth=1.8,
            label=f'After pruning (N={len(elev_after)/1e3:.0f}K)')

    ax.fill_between(c_b, d_b, d_a,
                    where=(d_b > d_a), alpha=0.15, color='red',    label='Removed')
    ax.fill_between(c_b, d_b, d_a,
                    where=(d_b <= d_a), alpha=0.15, color='green', label='Preserved')

    ax.set_xlabel('Elevation Angle (degrees)', fontsize=12)
    ax.set_ylabel('Normalized Density\n(relative to uniform sphere)', fontsize=10)
    ax.set_xlim(-90, 90)
    ax.set_xticks(range(-90, 91, 15))
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_title(f'{scene} — {label}: Pruning Before/After', fontsize=13)

    out = output_dir / f'gauss_dist_beforeafter_{scene}_{method_tag}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out}')


# ── 全シーン平均 ────────────────────────────────────────────────
def plot_all_scenes_avg(scenes: list[str], methods: list, output_dir: Path, bins: int = 36):
    all_elevations = {tag: [] for tag, _, _ in methods}

    for scene in scenes:
        for tag, _, _ in methods:
            d = find_exp_dir(scene, tag)
            if d is None:
                continue
            pos = load_positions_from_pt(d / 'checkpoints' / 'final.pt')
            if pos is None:
                continue
            all_elevations[tag].append(positions_to_elevation(pos))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    add_band_shading(ax)

    for tag, label, color in methods:
        if not all_elevations[tag]:
            continue
        combined = np.concatenate(all_elevations[tag])
        centers, density = elevation_to_density(combined, bins)
        ax.plot(centers, density, color=color, linewidth=1.8,
                label=f'{label} (total N={len(combined)/1e3:.0f}K)')

    ax.set_xlabel('Elevation Angle (degrees)', fontsize=12)
    ax.set_ylabel('Normalized Density (relative to uniform)', fontsize=10)
    ax.set_xlim(-90, 90)
    ax.set_xticks(range(-90, 91, 15))
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_title(f'Gaussian Elevation Distribution — avg. {len(scenes)} scenes (post-pruning)', fontsize=13)

    out = output_dir / 'gauss_dist_avg.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out}')


def main(scenes: list[str], output_dir: Path, before_after: bool):
    output_dir.mkdir(parents=True, exist_ok=True)

    for scene in scenes:
        print(f'\n--- {scene} ---')
        plot_method_comparison(scene, COMPARE_METHODS, output_dir)
        if before_after:
            for tag, label, color in COMPARE_METHODS:
                plot_before_after(scene, tag, label, color, output_dir)

    print('\n--- All scenes average ---')
    plot_all_scenes_avg(scenes, COMPARE_METHODS, output_dir)


if __name__ == '__main__':
    default_scenes = [
        'barbershop', 'archiviz-flat', 'bistro_bike', 'bistro_square',
        'classroom', 'fisher-hut', 'lone_monk', 'pavilion_midday_chair',
        'pavilion_midday_pond', 'restroom', 'LOU',
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes', nargs='+', default=default_scenes)
    parser.add_argument('--outdir', default='results/figures', type=Path)
    parser.add_argument('--before_after', action='store_true',
                        help='同一実験のbefore/after比較図も生成（positions_pre_pruning_*.npy が必要）')
    args = parser.parse_args()
    main(args.scenes, args.outdir, args.before_after)
