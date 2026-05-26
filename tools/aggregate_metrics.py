#!/usr/bin/env python3
"""metrics_8bit.txt を全実験から集約して手法別シーン平均表を作る。

output/SPaGS/*/test_30000/metrics_8bit.txt を読み込み、
PSNR / SSIM / LPIPS のシーン平均を CSV と見やすいテーブルで出力する。
"""

import re
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np


OUTPUT_ROOT = Path('output/SPaGS')
OUT_CSV     = Path('results/metrics_summary.csv')

# 集計対象メソッド（順番通りに表示）
METHODS_ORDER = [
    'spags', 'spags_opacity', 'spags_proposed',
    'baseline', 'opacity', 'proposed_v0', 'proposed_v1',
]

DATE_PAT = re.compile(r'_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$')
KNOWN_METHODS = [
    'spags_proposed', 'spags_opacity', 'spags',
    'proposed_v1', 'proposed_v0', 'opacity', 'baseline',
]


def parse_name(name: str) -> tuple[str, str] | None:
    base = DATE_PAT.sub('', name)
    for m in KNOWN_METHODS:
        if base.endswith('_' + m):
            scene = base[: -(len(m) + 1)]
            return scene, m
    return None


def parse_metrics(path: Path) -> dict | None:
    try:
        text = path.read_text()
    except Exception:
        return None
    psnr = re.search(r'PSNR\s+([\d.]+)', text)
    ssim = re.search(r'SSIM\s+([\d.]+)', text)
    lpips = re.search(r'LPIPS\s+([\d.]+)', text)
    if not (psnr and ssim and lpips):
        return None
    return {
        'psnr':  float(psnr.group(1)),
        'ssim':  float(ssim.group(1)),
        'lpips': float(lpips.group(1)),
    }


def main():
    rows = []
    for exp_dir in sorted(OUTPUT_ROOT.iterdir()):
        parsed = parse_name(exp_dir.name)
        if parsed is None:
            continue
        scene, method = parsed
        if method not in METHODS_ORDER:
            continue
        metrics_path = exp_dir / 'test_30000' / 'metrics_8bit.txt'
        m = parse_metrics(metrics_path)
        if m is None:
            continue
        # Gaussian数をcheckpoints/final.ptのサイズから取得（なければスキップ）
        n_gaussians = None
        final_pt = exp_dir / 'checkpoints' / 'final.pt'
        if final_pt.exists():
            try:
                import torch
                ckpt = torch.load(final_pt, map_location='cpu', weights_only=True)
                # positionsのshape[0]がGaussian数
                for key in ckpt:
                    if 'position' in key.lower() or '_positions' in key.lower():
                        n_gaussians = ckpt[key].shape[0]
                        break
            except Exception:
                pass
        rows.append({
            'scene': scene, 'method': method,
            'psnr': m['psnr'], 'ssim': m['ssim'], 'lpips': m['lpips'],
            'n_gaussians': n_gaussians,
        })

    if not rows:
        print('No results found.')
        return

    # CSV保存
    OUT_CSV.parent.mkdir(exist_ok=True)
    fieldnames = ['scene', 'method', 'psnr', 'ssim', 'lpips', 'n_gaussians']
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved → {OUT_CSV}  ({len(rows)} rows)\n')

    # シーン平均の集計
    by_method = defaultdict(list)
    for r in rows:
        by_method[r['method']].append(r)

    print(f'{"Method":<16} {"#Scenes":>7} {"PSNR":>8} {"SSIM":>8} {"LPIPS":>8} {"N_Gauss":>10}')
    print('─' * 62)
    for method in METHODS_ORDER:
        if method not in by_method:
            continue
        vals = by_method[method]
        psnr_avg  = np.mean([v['psnr']  for v in vals])
        ssim_avg  = np.mean([v['ssim']  for v in vals])
        lpips_avg = np.mean([v['lpips'] for v in vals])
        n_gauss   = [v['n_gaussians'] for v in vals if v['n_gaussians']]
        ng_str    = f'{np.mean(n_gauss):>10,.0f}' if n_gauss else '         -'
        print(f'{method:<16} {len(vals):>7} {psnr_avg:>8.3f} {ssim_avg:>8.4f} {lpips_avg:>8.4f} {ng_str}')

    # シーン別詳細
    print(f'\n{"Scene":<28} {"Method":<16} {"PSNR":>8} {"SSIM":>8} {"LPIPS":>8} {"N_Gauss":>10}')
    print('─' * 82)
    scenes = sorted({r['scene'] for r in rows})
    for scene in scenes:
        for method in METHODS_ORDER:
            match = [r for r in rows if r['scene'] == scene and r['method'] == method]
            if not match:
                continue
            r = match[0]
            ng = f'{r["n_gaussians"]:>10,}' if r['n_gaussians'] else '         -'
            print(f'{scene:<28} {method:<16} {r["psnr"]:>8.3f} {r["ssim"]:>8.4f} {r["lpips"]:>8.4f} {ng}')
        print()


if __name__ == '__main__':
    main()
