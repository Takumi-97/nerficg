#!/usr/bin/env python3
"""距離別Gaussian分布の可視化。

使い方:
  python tools/visualize_gaussian_distance.py --scene barbershop
  python tools/visualize_gaussian_distance.py --all-scenes
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_positions(scene: str, method: str) -> np.ndarray | None:
    root = Path('output/SPaGS')
    for d in sorted(root.glob(f'{scene}_{method}_*'), reverse=True):
        p = d / 'checkpoints' / 'final.pt'
        if p.exists():
            ckpt = torch.load(p, map_location='cpu', weights_only=False)
            return ckpt['model_state_dict']['gaussians._positions'].numpy()
    return None


METHODS = {
    'v2_baseline': ('Baseline (no pruning)', '#2196F3', '-',  2.0),
    'v2_opacity':  ('Opacity pruning',       '#FF5722', '--', 1.5),
    'v2_proposed': ('Proposed (region-norm)','#4CAF50', '--', 1.5),
}


def plot_distance_hist(scenes: list[str], out: Path):
    """log scaleヒストグラム + 分位数別保持率。"""
    n = len(scenes)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, scene in enumerate(scenes):
        ax_log = axes[row, 0]
        ax_bar = axes[row, 1]

        base_pos = load_positions(scene, 'v2_baseline')
        if base_pos is None:
            ax_log.set_title(f'{scene} (not found)')
            continue

        center = base_pos.mean(axis=0)
        base_dists = np.linalg.norm(base_pos - center, axis=1)

        # log-scale bins
        d_min = max(base_dists.min(), 0.01)
        d_max = base_dists.max()
        bins = np.logspace(np.log10(d_min), np.log10(d_max), 60)

        # 分位数境界（遠景判定用）
        p_far = np.percentile(base_dists, 70)
        p90   = np.percentile(base_dists, 90)
        p99   = np.percentile(base_dists, 99)

        # ── 左: log-scaleヒストグラム ──
        for method, (label, color, ls, lw) in METHODS.items():
            pos = load_positions(scene, method)
            if pos is None:
                continue
            dists = np.linalg.norm(pos - center, axis=1)
            counts, _ = np.histogram(dists, bins=bins)
            bin_c = np.sqrt(bins[:-1] * bins[1:])  # 対数中点
            ax_log.plot(bin_c, counts, color=color, linestyle=ls, linewidth=lw,
                        label=f'{label} (N={len(pos):,})')

        ax_log.axvline(p_far, color='red',    ls='-.', lw=1.0, alpha=0.8, label=f'70%ile={p_far:.1f}')
        ax_log.axvline(p90,   color='orange', ls=':',  lw=1.0, alpha=0.8, label=f'90%ile={p90:.1f}')
        ax_log.axvline(p99,   color='purple', ls=':',  lw=1.0, alpha=0.8, label=f'99%ile={p99:.1f}')
        ax_log.set_xscale('log')
        ax_log.set_xlabel('Distance from scene center (log scale)')
        ax_log.set_ylabel('Number of Gaussians')
        ax_log.set_title(f'{scene}')
        ax_log.legend(fontsize=7)
        ax_log.grid(True, alpha=0.3, which='both')

        # ── 右: 距離区間別 保持率 ──
        # 区間: [0, p70), [p70, p90), [p90, p99), [p99, ∞)
        bin_edges = [0, p_far, p90, p99, np.inf]
        bin_labels = ['<p70\n(近景)', 'p70-p90\n(中景)', 'p90-p99\n(遠景)', '>p99\n(極遠)']
        n_bins = len(bin_labels)

        base_counts = np.array([
            ((base_dists >= bin_edges[i]) & (base_dists < bin_edges[i+1])).sum()
            for i in range(n_bins)
        ])

        x = np.arange(n_bins)
        width = 0.35
        method_list = ['v2_opacity', 'v2_proposed']
        for mi, method in enumerate(method_list):
            pos = load_positions(scene, method)
            if pos is None:
                continue
            dists = np.linalg.norm(pos - center, axis=1)
            counts = np.array([
                ((dists >= bin_edges[i]) & (dists < bin_edges[i+1])).sum()
                for i in range(n_bins)
            ])
            retain = np.where(base_counts > 0, counts / base_counts * 100, 0)
            label, color, _, _ = METHODS[method]
            offset = (mi - 0.5) * width
            bars = ax_bar.bar(x + offset, retain, width, label=label, color=color, alpha=0.8)
            for bar, val in zip(bars, retain):
                if val > 2:
                    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                f'{val:.0f}%', ha='center', va='bottom', fontsize=7)

        ax_bar.axhline(100, color='black', ls='--', lw=0.8, alpha=0.4)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(bin_labels, fontsize=9)
        ax_bar.set_ylabel('Retention vs baseline (%)')
        ax_bar.set_ylim(0, 115)
        ax_bar.set_title(f'{scene} — 距離区間別 保持率')
        ax_bar.legend(fontsize=8)
        ax_bar.grid(True, axis='y', alpha=0.3)

        # 各区間のbaseline絶対数を注釈
        for i, (cnt, label) in enumerate(zip(base_counts, bin_labels)):
            ax_bar.text(i, 108, f'N={cnt:,}', ha='center', fontsize=7, color='gray')

    fig.suptitle('Gaussian distance distribution & retention by region', fontsize=13)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='barbershop')
    parser.add_argument('--all-scenes', action='store_true')
    parser.add_argument('--out', default='results/gaussian_distance_dist.png')
    args = parser.parse_args()

    scenes = [
        'barbershop', 'archiviz-flat', 'bistro_bike', 'bistro_square',
        'classroom', 'fisher-hut', 'lone_monk', 'pavilion_midday_chair',
        'pavilion_midday_pond', 'restroom',
    ] if args.all_scenes else [args.scene]

    plot_distance_hist(scenes, Path(args.out))


if __name__ == '__main__':
    main()
