#!/usr/bin/env python3
"""アブレーション棒グラフ生成。

results/metrics_summary.csv から読み込み、
各コンポーネントの寄与をシーン平均 PSNR/SSIM/LPIPS で可視化する。
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── メソッド定義 ─────────────────────────────────────────────────────
ABLATION_STEPS = [
    # (method_key, display_label, color)
    ('spags',          'SPaGS\n(Original)',         '#888888'),
    ('baseline',       'Baseline\n(+WS-L1)',         '#4477AA'),
    ('opacity',        'Opacity\nPruning',           '#66AADD'),
    ('proposed_v0',    'Proposed\n(w/o SA-Op)',       '#EE8833'),
    ('proposed_v1',    'Proposed\n(full)',             '#E05A2B'),
]

SPAGS_VARIANTS = [
    ('spags',          'SPaGS',                     '#888888'),
    ('spags_opacity',  'SPaGS\n+Opacity',           '#AAAAAA'),
    ('spags_proposed', 'SPaGS\n+Proposed',          '#CC4411'),
]


def load_csv(path: Path) -> list[dict]:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def scene_avg(rows: list[dict], method: str) -> dict | None:
    vals = [r for r in rows if r['method'] == method]
    if not vals:
        return None
    return {
        'psnr':  np.mean([float(v['psnr'])  for v in vals]),
        'ssim':  np.mean([float(v['ssim'])  for v in vals]),
        'lpips': np.mean([float(v['lpips']) for v in vals]),
        'n':     len(vals),
    }


def plot_bars(steps: list[tuple], rows: list[dict], metric: str,
              output: Path, title: str, ref_method: str | None = None):
    methods = [s[0] for s in steps]
    labels  = [s[1] for s in steps]
    colors  = [s[2] for s in steps]

    avgs = [scene_avg(rows, m) for m in methods]
    vals = [a[metric] if a else np.nan for a in avgs]

    fig, ax = plt.subplots(figsize=(len(steps) * 1.6 + 1.5, 4.5))

    bars = ax.bar(range(len(steps)), vals, color=colors,
                  edgecolor='white', linewidth=0.5, width=0.65)

    # 値ラベル
    for bar, v, a in zip(bars, vals, avgs):
        if np.isnan(v):
            continue
        n_str = f'n={a["n"]}' if a else ''
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{v:.3f}\n({n_str})',
                ha='center', va='bottom', fontsize=8.5)

    # 差分矢印（ref_methodがある場合）
    if ref_method and ref_method in methods:
        ref_idx = methods.index(ref_method)
        ref_val = vals[ref_idx]
        for i, (v, m) in enumerate(zip(vals, methods)):
            if i == ref_idx or np.isnan(v):
                continue
            diff = v - ref_val
            sign = '+' if diff >= 0 else ''
            ax.annotate(
                f'{sign}{diff:.3f}',
                xy=(i, v + 0.05),
                ha='center', va='bottom',
                fontsize=8, color='#D04000' if diff > 0 else '#0044AA',
            )

    # 基準線（ref_method）
    if ref_method and ref_method in methods:
        ref_val = vals[methods.index(ref_method)]
        if not np.isnan(ref_val):
            ax.axhline(ref_val, color='#888888', linestyle='--',
                       linewidth=1, alpha=0.6, label=f'Ref: {ref_method}')

    ylabel = {'psnr': 'PSNR (dB)', 'ssim': 'SSIM', 'lpips': 'LPIPS'}.get(metric, metric)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    if title:
        ax.set_title(title, fontsize=13)

    valid = [v for v in vals if not np.isnan(v)]
    if valid:
        margin = (max(valid) - min(valid)) * 0.3 + 0.3
        ax.set_ylim(min(valid) - margin, max(valid) + margin + 0.3)

    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {output}')


def main(csv_path: Path, output_dir: Path):
    rows = load_csv(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in ['psnr', 'ssim', 'lpips']:
        metric_label = metric.upper()
        # アブレーション（ours系列）
        plot_bars(
            ABLATION_STEPS, rows, metric,
            output_dir / f'ablation_{metric}.png',
            title=f'Ablation Study — {metric_label} (avg. all scenes)',
            ref_method='spags',
        )
        # SPaGS系列との比較
        plot_bars(
            SPAGS_VARIANTS, rows, metric,
            output_dir / f'spags_variants_{metric}.png',
            title=f'SPaGS Variants — {metric_label} (avg. all scenes)',
            ref_method='spags',
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',    default='results/metrics_summary.csv', type=Path)
    parser.add_argument('--outdir', default='results/figures',             type=Path)
    args = parser.parse_args()
    main(args.csv, args.outdir)
