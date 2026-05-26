#!/usr/bin/env python3
"""緯度帯別PSNR/SSIMの棒グラフを生成。

results/latitude_eval.csv から読み込み、
極帯/中緯度帯/赤道帯ごとにメソッド比較を描画する。
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METHOD_LABEL = {
    'spags':          'SPaGS',
    'baseline':       'Baseline (WS-L1)',
    'opacity':        'Opacity Pruning',
    'proposed_v0':    'Proposed (w/o SA)',
    'proposed_v1':    'Proposed (full)',
    'spags_opacity':  'SPaGS+Opacity',
    'spags_proposed': 'SPaGS+Proposed',
}

METHOD_COLOR = {
    'spags':          '#888888',
    'baseline':       '#4477AA',
    'opacity':        '#66AADD',
    'proposed_v0':    '#EE8833',
    'proposed_v1':    '#E05A2B',
    'spags_opacity':  '#AAAAAA',
    'spags_proposed': '#CC4411',
}

BAND_LABEL = {
    'polar':      'Polar (|θ|>60°)',
    'mid':        'Mid (30°<|θ|<60°)',
    'equatorial': 'Equatorial (|θ|<30°)',
}

METHODS_ORDER = ['spags', 'baseline', 'opacity', 'proposed_v0', 'proposed_v1']


def load_csv(path: Path) -> list[dict]:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def aggregate_by_band(rows: list[dict], methods: list[str]) -> dict:
    """method → band → avg PSNR/SSIM。"""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        m = r['method']
        b = r['band']
        if m not in methods or b == 'full':
            continue
        acc[m][b].append({'psnr': float(r['psnr']), 'ssim': float(r['ssim'])})

    result = {}
    for m, by_band in acc.items():
        result[m] = {}
        for b, vals in by_band.items():
            result[m][b] = {
                'psnr': np.mean([v['psnr'] for v in vals]),
                'ssim': np.mean([v['ssim'] for v in vals]),
            }
    return result


def plot_band_comparison(agg: dict, metric: str, methods: list[str],
                         output: Path, title: str = ''):
    bands = ['polar', 'mid', 'equatorial']
    present_methods = [m for m in methods if m in agg]
    n_methods = len(present_methods)
    if n_methods == 0:
        print(f'  SKIP: no data for {output}')
        return
    n_bands = len(bands)

    x = np.arange(n_bands)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i, method in enumerate(present_methods):
        vals = [agg[method].get(b, {}).get(metric, np.nan) for b in bands]
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      label=METHOD_LABEL.get(method, method),
                      color=METHOD_COLOR.get(method, '#999999'),
                      edgecolor='white', linewidth=0.5)
        # 値ラベル
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.05,
                        f'{v:.2f}', ha='center', va='bottom',
                        fontsize=6.5, rotation=90, color='#333333')

    ylabel = 'PSNR (dB)' if metric == 'psnr' else 'SSIM'
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([BAND_LABEL[b] for b in bands], fontsize=11)
    ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    if title:
        ax.set_title(title, fontsize=13)

    # y軸の下限を自動調整（値の最小値 - 1 dB）
    valid = [v for m in present_methods
             for b in bands
             for v in [agg[m].get(b, {}).get(metric, np.nan)]
             if not np.isnan(v)]
    if valid:
        ymin = min(valid) - 1.5
        ymax = max(valid) + 2.0
        ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {output}')


def main(csv_path: Path, output_dir: Path, methods: list[str]):
    rows = load_csv(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    agg = aggregate_by_band(rows, methods)
    present = [m for m in methods if m in agg]
    print(f'Methods found: {present}')

    for metric in ['psnr', 'ssim']:
        label = 'PSNR' if metric == 'psnr' else 'SSIM'
        plot_band_comparison(
            agg, metric, methods,
            output_dir / f'latitude_{metric}.png',
            title=f'Latitude Band {label} Comparison (avg. all scenes)',
        )

    # シーン別も出力
    scenes = sorted({r['scene'] for r in rows})
    for scene in scenes:
        scene_rows = [r for r in rows if r['scene'] == scene]
        agg_s = aggregate_by_band(scene_rows, methods)
        plot_band_comparison(
            agg_s, 'psnr', methods,
            output_dir / f'latitude_{scene}_psnr.png',
            title=f'{scene} — Latitude Band PSNR',
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',    default='results/latitude_eval.csv', type=Path)
    parser.add_argument('--outdir', default='results/figures',           type=Path)
    parser.add_argument('--methods', nargs='+', default=METHODS_ORDER)
    args = parser.parse_args()
    main(args.csv, args.outdir, args.methods)
