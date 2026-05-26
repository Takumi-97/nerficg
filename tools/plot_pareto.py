#!/usr/bin/env python3
"""Pareto曲線プロット。results/pareto_eval.csv から生成。

使用例:
  python tools/plot_pareto.py
  python tools/plot_pareto.py --output results/pareto.pdf
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


METHOD_STYLE = {
    'no_pruning': dict(color='#444444', linestyle='--',  marker='D', label='No Pruning (baseline)'),
    'proposed':   dict(color='#E05A2B', linestyle='-',   marker='o', label='Proposed (region-normalized)'),
    'opacity':    dict(color='#2B7BE0', linestyle='-',   marker='s', label='Opacity Pruning'),
}


def load_csv(path: Path) -> list[dict]:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def aggregate(rows: list[dict]) -> dict:
    """シーン平均の (n_gaussians, psnr) を method ごとに返す。"""
    # keep_ratio ごとに各シーンの値を収集
    by_method_kr = defaultdict(lambda: defaultdict(list))
    for r in rows:
        method = r['method']
        kr     = float(r['keep_ratio'])
        by_method_kr[method][kr].append({
            'n': int(r['n_gaussians']),
            'psnr': float(r['psnr']),
            'ssim': float(r['ssim']),
        })

    result = {}
    for method, by_kr in by_method_kr.items():
        points = []
        for kr in sorted(by_kr.keys()):
            vals = by_kr[kr]
            points.append({
                'keep_ratio': kr,
                'n_gaussians': np.mean([v['n'] for v in vals]),
                'psnr':        np.mean([v['psnr'] for v in vals]),
                'ssim':        np.mean([v['ssim'] for v in vals]),
                'n_scenes':    len(vals),
            })
        result[method] = points
    return result


def plot(agg: dict, x_key: str, y_key: str, output: Path, title: str = ''):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for method, points in agg.items():
        style = METHOD_STYLE.get(method, {})
        xs = [p[x_key] for p in points]
        ys = [p[y_key] for p in points]

        # no_pruningは点のみ
        if method == 'no_pruning':
            ax.scatter(xs, ys,
                       color=style['color'], marker=style['marker'],
                       s=80, zorder=5, label=style['label'])
        else:
            ax.plot(xs, ys,
                    color=style['color'], linestyle=style['linestyle'],
                    marker=style['marker'], markersize=6,
                    linewidth=1.8, label=style['label'])

        # keep_ratioのアノテーション（proposed のみ）
        if method == 'proposed':
            for p in points:
                ax.annotate(
                    f"{p['keep_ratio']:.1f}",
                    xy=(p[x_key], p[y_key]),
                    xytext=(4, 4), textcoords='offset points',
                    fontsize=7, color=style['color'], alpha=0.8,
                )

    xlabel = 'Number of Gaussians' if x_key == 'n_gaussians' else x_key
    ylabel = 'PSNR (dB)' if y_key == 'psnr' else 'SSIM'

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K'))
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.4)
    if title:
        ax.set_title(title, fontsize=13)

    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {output}')


def main(csv_path: Path, output_dir: Path):
    rows = load_csv(csv_path)
    scenes = sorted({r['scene'] for r in rows})
    n_scenes = len(scenes)
    print(f'Loaded {len(rows)} rows, {n_scenes} scenes: {scenes}')

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 全シーン平均 ──
    agg = aggregate(rows)
    plot(agg, 'n_gaussians', 'psnr',
         output_dir / 'pareto_avg_psnr.png',
         title=f'Quality-Compression Tradeoff (avg. {n_scenes} scenes)')
    plot(agg, 'n_gaussians', 'ssim',
         output_dir / 'pareto_avg_ssim.png',
         title=f'Quality-Compression Tradeoff (avg. {n_scenes} scenes)')

    # ── シーン別 ──
    for scene in scenes:
        scene_rows = [r for r in rows if r['scene'] == scene]
        agg_s = aggregate(scene_rows)
        plot(agg_s, 'n_gaussians', 'psnr',
             output_dir / f'pareto_{scene}_psnr.png',
             title=f'{scene}')

    # ── サマリー表示 ──
    print(f'\n{"Method":<20} {"keep":<6} {"N_Gauss (avg)":>14} {"PSNR (avg)":>11} {"n_scenes":>9}')
    print('-' * 65)
    for method in ['no_pruning', 'proposed', 'opacity']:
        if method not in agg:
            continue
        for p in agg[method]:
            print(f'{method:<20} {p["keep_ratio"]:<6.1f} {p["n_gaussians"]:>14,.0f} {p["psnr"]:>11.3f} {p["n_scenes"]:>9}')
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',    default='results/pareto_eval.csv', type=Path)
    parser.add_argument('--outdir', default='results/pareto_plots',    type=Path)
    args = parser.parse_args()
    main(args.csv, args.outdir)
