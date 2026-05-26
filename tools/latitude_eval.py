#!/usr/bin/env python3
"""緯度帯別メトリクス評価スクリプト。

test_30000/rgb/ と test_30000/rgb_gt/ が存在する実験ディレクトリを
自動スキャンし、全テスト画像の PSNR / SSIM を緯度帯別に計算して CSV に保存する。

緯度帯（equirectangular H=1000 基準）:
  polar     : |θ| > 60°  → 上1/6 + 下1/6
  mid       : 30° < |θ| < 60° → その内側
  equatorial: |θ| < 30°  → 中央1/3
"""

import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ── 緯度帯定義 ─────────────────────────────────────────────
BANDS = {
    'polar':      (60.0,  90.0),
    'mid':        (30.0,  60.0),
    'equatorial': ( 0.0,  30.0),
}


def lat_row_slice(H: int, lat_min_deg: float, lat_max_deg: float) -> np.ndarray:
    """|θ| が [lat_min, lat_max] に入る行インデックスを返す。"""
    i = np.arange(H)
    theta_deg = np.abs((0.5 - (i + 0.5) / H) * 180.0)
    return np.where((theta_deg >= lat_min_deg) & (theta_deg < lat_max_deg))[0]


def load_image(path: Path) -> torch.Tensor:
    """PNG を [3, H, W] float32 tensor (0~1) で返す。"""
    img = np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = F.mse_loss(pred, gt).item()
    return float('inf') if mse < 1e-10 else 10 * np.log10(1.0 / mse)


def ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    p = pred.unsqueeze(0)
    g = gt.unsqueeze(0)
    mu1 = F.avg_pool2d(p, 3, 1, 1)
    mu2 = F.avg_pool2d(g, 3, 1, 1)
    s1  = F.avg_pool2d(p * p, 3, 1, 1) - mu1 ** 2
    s2  = F.avg_pool2d(g * g, 3, 1, 1) - mu2 ** 2
    s12 = F.avg_pool2d(p * g, 3, 1, 1) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    val = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / \
          ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
    return val.mean().item()


def eval_dir(rgb_dir: Path, gt_dir: Path) -> dict:
    """1実験の全テスト画像を緯度帯別に評価して結果 dict を返す。"""
    files = sorted(rgb_dir.glob('*.png'))
    if not files:
        return {}

    H = load_image(files[0]).shape[1]
    band_rows = {name: lat_row_slice(H, lo, hi) for name, (lo, hi) in BANDS.items()}

    accum = {name: {'psnr': [], 'ssim': []} for name in BANDS}
    accum['full'] = {'psnr': [], 'ssim': []}

    for f in files:
        gt_path = gt_dir / f.name
        if not gt_path.exists():
            continue
        try:
            pred = load_image(f)
            gt   = load_image(gt_path)
        except Exception:
            continue

        accum['full']['psnr'].append(psnr(pred, gt))
        accum['full']['ssim'].append(ssim(pred, gt))

        for name, rows in band_rows.items():
            p_band = pred[:, rows, :]
            g_band = gt[:, rows, :]
            accum[name]['psnr'].append(psnr(p_band, g_band))
            accum[name]['ssim'].append(ssim(p_band, g_band))

    return {
        band: {
            'psnr': float(np.mean(vals['psnr'])),
            'ssim': float(np.mean(vals['ssim'])),
            'n':    len(vals['psnr']),
        }
        for band, vals in accum.items()
    }


def parse_exp_name(name: str) -> tuple[str, str]:
    """
    'fisher-hut_spags_proposed_2026-05-21-12-06-07'
    → scene='fisher-hut', method='spags_proposed'
    """
    date_pat = re.compile(r'_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$')
    base = date_pat.sub('', name)
    # 既知のメソッド名（長い順にマッチさせる）
    methods = [
        'spags_proposed', 'spags_opacity', 'spags',
        'proposed_v1', 'proposed_v0', 'opacity', 'baseline',
    ]
    for m in methods:
        if base.endswith('_' + m):
            scene = base[: -(len(m) + 1)]
            return scene, m
    # マッチしない場合は最後の '_' 以降をメソッドとみなす
    parts = base.rsplit('_', 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (base, 'unknown')


def main():
    output_root = Path('output/SPaGS')
    out_csv = Path('results/latitude_eval.csv')
    out_csv.parent.mkdir(exist_ok=True)

    rows = []
    exp_dirs = sorted(output_root.iterdir())

    for exp_dir in exp_dirs:
        rgb_dir = exp_dir / 'test_30000' / 'rgb'
        gt_dir  = exp_dir / 'test_30000' / 'rgb_gt'
        if not rgb_dir.exists() or not gt_dir.exists():
            continue

        scene, method = parse_exp_name(exp_dir.name)
        print(f'Evaluating {scene} / {method} ...', end=' ', flush=True)

        result = eval_dir(rgb_dir, gt_dir)
        if not result:
            print('no images, skip')
            continue

        for band, metrics in result.items():
            rows.append({
                'scene':  scene,
                'method': method,
                'band':   band,
                'psnr':   round(metrics['psnr'], 4),
                'ssim':   round(metrics['ssim'], 4),
                'n_imgs': metrics['n'],
            })
        print(f"done ({result['full']['n']} images)")

    if not rows:
        print('No results found.')
        sys.exit(1)

    # CSV 書き出し
    import csv
    fieldnames = ['scene', 'method', 'band', 'psnr', 'ssim', 'n_imgs']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nSaved → {out_csv}  ({len(rows)} rows)')

    # サマリー表示
    import collections
    by_method_band = collections.defaultdict(list)
    for r in rows:
        by_method_band[(r['method'], r['band'])].append(r['psnr'])

    bands_order = ['polar', 'mid', 'equatorial', 'full']
    methods_order = ['spags', 'spags_opacity', 'spags_proposed',
                     'baseline', 'opacity', 'proposed_v0', 'proposed_v1']

    print(f'\n{"Method":<20} {"Band":<12} {"PSNR (avg over scenes)":>22}')
    print('-' * 56)
    for method in methods_order:
        for band in bands_order:
            key = (method, band)
            if key in by_method_band:
                avg = np.mean(by_method_band[key])
                print(f'{method:<20} {band:<12} {avg:>22.4f}')
        print()


if __name__ == '__main__':
    main()
