#!/usr/bin/env python3
"""Barbershop ablation: 簡略版proposed に4要素を段階的に追加して影響を検証"""

import csv
import glob
import re
import subprocess
import tempfile
import time
import yaml
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

SCENE = 'barbershop'
WANDB_PROJECT = 'OmniPrune'
WANDB_ENTITY = None

RESULTS_CSV = Path('results/barbershop_ablation.csv')
RESULTS_CSV.parent.mkdir(exist_ok=True)

# ── アブレーション設定 ─────────────────────────────────────
# ベース: proposed (simplified) = 仰角補正 + region_normalized のみ
# 下記は累積的に要素を追加（①→②→③→④でフル実装になる）
ABLATION_METHODS = {
    'proposed_v0': {
        'desc': 'baseline: 仰角補正 + region_normalized のみ (簡略版)',
        'flags': {
            'USE_CONTRIBUTION_PRUNING':        True,
            'USE_OPACITY_BASELINE':            False,
            'USE_OPACITY_SCORE':               False,
            'USE_UNIFORM_CONTRIBUTION':        False,
            'USE_REGION_NORMALIZED':           True,
            'CONTRIBUTION_PRUNING_USE_VOLUME': False,
            'USE_ERROR_WEIGHT':                False,
            'SA_OPACITY_PRUNING':              False,
            'USE_LAT_DENSIFY_CORRECTION':      False,
            'EXPERIMENT_TAG':                  'proposed_v0',
        },
    },
    'proposed_v1_sa': {
        'desc': '+ SA_OPACITY_PRUNING',
        'flags': {
            'USE_CONTRIBUTION_PRUNING':        True,
            'USE_OPACITY_BASELINE':            False,
            'USE_OPACITY_SCORE':               False,
            'USE_UNIFORM_CONTRIBUTION':        False,
            'USE_REGION_NORMALIZED':           True,
            'CONTRIBUTION_PRUNING_USE_VOLUME': False,
            'USE_ERROR_WEIGHT':                False,
            'SA_OPACITY_PRUNING':              True,   # ← 追加
            'USE_LAT_DENSIFY_CORRECTION':      False,
            'EXPERIMENT_TAG':                  'proposed_v1_sa',
        },
    },
    'proposed_v2_lat': {
        'desc': '+ SA_OPACITY_PRUNING + USE_LAT_DENSIFY_CORRECTION',
        'flags': {
            'USE_CONTRIBUTION_PRUNING':        True,
            'USE_OPACITY_BASELINE':            False,
            'USE_OPACITY_SCORE':               False,
            'USE_UNIFORM_CONTRIBUTION':        False,
            'USE_REGION_NORMALIZED':           True,
            'CONTRIBUTION_PRUNING_USE_VOLUME': False,
            'USE_ERROR_WEIGHT':                False,
            'SA_OPACITY_PRUNING':              True,
            'USE_LAT_DENSIFY_CORRECTION':      True,   # ← 追加
            'EXPERIMENT_TAG':                  'proposed_v2_lat',
        },
    },
    'proposed_v3_vol': {
        'desc': '+ SA_OPACITY + LAT_DENSIFY + VOLUME',
        'flags': {
            'USE_CONTRIBUTION_PRUNING':        True,
            'USE_OPACITY_BASELINE':            False,
            'USE_OPACITY_SCORE':               False,
            'USE_UNIFORM_CONTRIBUTION':        False,
            'USE_REGION_NORMALIZED':           True,
            'CONTRIBUTION_PRUNING_USE_VOLUME': True,   # ← 追加
            'USE_ERROR_WEIGHT':                False,
            'SA_OPACITY_PRUNING':              True,
            'USE_LAT_DENSIFY_CORRECTION':      True,
            'EXPERIMENT_TAG':                  'proposed_v3_vol',
        },
    },
    'proposed_v4_full': {
        'desc': '全要素あり (フル実装復元)',
        'flags': {
            'USE_CONTRIBUTION_PRUNING':        True,
            'USE_OPACITY_BASELINE':            False,
            'USE_OPACITY_SCORE':               False,
            'USE_UNIFORM_CONTRIBUTION':        False,
            'USE_REGION_NORMALIZED':           True,
            'CONTRIBUTION_PRUNING_USE_VOLUME': True,
            'USE_ERROR_WEIGHT':                True,   # ← 追加
            'ERROR_WEIGHT_LAMBDA':             2.0,
            'SA_OPACITY_PRUNING':              True,
            'USE_LAT_DENSIFY_CORRECTION':      True,
            'EXPERIMENT_TAG':                  'proposed_v4_full',
        },
    },
}


# ── 評価ユーティリティ ────────────────────────────────────

def ws_psnr(pred_dir: Path, gt_dir: Path) -> float:
    preds = sorted(pred_dir.glob('*.png'))
    gts   = sorted(gt_dir.glob('*.png'))
    assert len(preds) == len(gts) > 0
    total_ws_mse = total_weight = 0.0
    for p, g in zip(preds, gts):
        pred = np.array(Image.open(p)).astype(np.float64) / 255.0
        gt   = np.array(Image.open(g)).astype(np.float64) / 255.0
        H = pred.shape[0]
        w = np.sin(np.pi * (np.arange(H) + 0.5) / H)[:, None, None]
        total_ws_mse += np.mean(w * (pred - gt) ** 2)
        total_weight += np.mean(w)
    mse_norm = (total_ws_mse / len(preds)) / (total_weight / len(preds))
    return -10 * np.log10(mse_norm) if mse_norm > 0 else float('inf')


def read_metrics(output_dir: Path) -> dict | None:
    test_dir = output_dir / 'test_30000'
    if not test_dir.exists():
        return None
    txt = (test_dir / 'metrics_8bit.txt').read_text()

    psnr  = float(re.search(r'^PSNR\t([\d.]+)',  txt, re.M).group(1))
    ssim  = float(re.search(r'^SSIM\t([\d.]+)',  txt, re.M).group(1))
    lpips = float(re.search(r'^LPIPS\t([\d.]+)', txt, re.M).group(1))
    wsp   = ws_psnr(test_dir / 'rgb', test_dir / 'rgb_gt')

    ckpt_files = sorted((output_dir / 'checkpoints').glob('final.pt'))
    n_pts = None
    if ckpt_files:
        ckpt = torch.load(ckpt_files[0], map_location='cpu', weights_only=False)
        n_pts = ckpt['model_state_dict']['gaussians._positions'].shape[0]

    return {'psnr': psnr, 'ws_psnr': wsp, 'ssim': ssim, 'lpips': lpips, 'n_points': n_pts}


def find_latest_output(method_name: str) -> Path | None:
    candidates = sorted(Path('output/SPaGS').glob(f'{SCENE}_{method_name}_*'), reverse=True)
    for d in candidates:
        if (d / 'checkpoints' / 'final.pt').exists():
            return d
    return None


# ── CSV 記録 ────────────────────────────────────────────

CSV_HEADER = ['method', 'desc', 'psnr', 'ws_psnr', 'ssim', 'lpips', 'n_points', 'elapsed_min']

def init_csv():
    if not RESULTS_CSV.exists():
        with open(RESULTS_CSV, 'w', newline='') as f:
            csv.writer(f).writerow(CSV_HEADER)

def append_csv(row: dict):
    with open(RESULTS_CSV, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([row.get(k, '') for k in CSV_HEADER])

def load_csv() -> list[dict]:
    if not RESULTS_CSV.exists():
        return []
    with open(RESULTS_CSV) as f:
        return list(csv.DictReader(f))


# ── 既存結果の取り込み ─────────────────────────────────────

REFERENCE_METHODS = {
    'barbershop_baseline': 'baseline (pruningなし)',
    'barbershop_opacity':  'opacity pruning (ベースライン手法)',
}

def record_reference_results():
    rows = load_csv()
    existing = {r['method'] for r in rows}
    for method_key, desc in REFERENCE_METHODS.items():
        if method_key in existing:
            continue
        d = find_latest_output(method_key.replace('barbershop_', ''))
        if d is None:
            print(f'[SKIP] {method_key}: output not found')
            continue
        m = read_metrics(d)
        if m is None:
            print(f'[SKIP] {method_key}: metrics not found')
            continue
        row = {'method': method_key, 'desc': desc, 'elapsed_min': '', **m}
        append_csv(row)
        print(f'[REF] {method_key}: PSNR={m["psnr"]:.2f}, WS-PSNR={m["ws_psnr"]:.2f}, '
              f'SSIM={m["ssim"]:.4f}, LPIPS={m["lpips"]:.4f}, pts={m["n_points"]:,}')


# ── 実験実行 ───────────────────────────────────────────────

def load_base_config(scene: str) -> dict:
    with open(f'configs/{scene}.yaml') as f:
        return yaml.safe_load(f)


def run_experiment(method_name: str, method_info: dict) -> dict | None:
    rows = load_csv()
    existing = {r['method'] for r in rows}
    full_key = f'{SCENE}_{method_name}'
    if full_key in existing:
        print(f'[SKIP] {full_key}: already recorded in CSV')
        return None

    config = load_base_config(SCENE)
    config['TRAINING']['WANDB']['ACTIVATE'] = True
    config['TRAINING']['WANDB']['PROJECT'] = WANDB_PROJECT
    if WANDB_ENTITY is not None:
        config['TRAINING']['WANDB']['ENTITY'] = WANDB_ENTITY
    config['TRAINING']['MODEL_NAME'] = full_key

    for key, val in method_info['flags'].items():
        config['TRAINING'][key] = val

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        temp_path = f.name

    print(f'\n{"="*60}')
    print(f'  Method: {method_name}')
    print(f'  Desc  : {method_info["desc"]}')
    print(f'{"="*60}')

    t0 = time.time()
    try:
        subprocess.run(['python', 'scripts/train.py', '-c', temp_path], check=True)
        elapsed = (time.time() - t0) / 60
        print(f'  ✓ Done in {elapsed:.1f} min')
    except subprocess.CalledProcessError as e:
        print(f'  ✗ FAILED: {e}')
        return None
    finally:
        os.unlink(temp_path)

    d = find_latest_output(method_name)
    if d is None:
        print(f'  ✗ Output directory not found')
        return None

    m = read_metrics(d)
    if m is None:
        print(f'  ✗ Metrics not found')
        return None

    row = {'method': full_key, 'desc': method_info['desc'], 'elapsed_min': f'{elapsed:.1f}', **m}
    append_csv(row)
    print(f'  → PSNR={m["psnr"]:.2f}, WS-PSNR={m["ws_psnr"]:.2f}, '
          f'SSIM={m["ssim"]:.4f}, LPIPS={m["lpips"]:.4f}, pts={m["n_points"]:,}')
    return m


# ── サマリー表示 ───────────────────────────────────────────

def print_summary():
    rows = load_csv()
    if not rows:
        print('No results yet.')
        return

    print(f'\n{"="*80}')
    print(f'  Barbershop Ablation Results')
    print(f'{"="*80}')
    print(f'{"Method":<30} {"PSNR":>6} {"WS-PSNR":>8} {"SSIM":>6} {"LPIPS":>6} {"Points":>8}')
    print(f'{"-"*70}')
    for r in rows:
        pts = int(r['n_points']) if r.get('n_points') else 0
        print(f'{r["method"]:<30} {float(r["psnr"]):>6.2f} {float(r["ws_psnr"]):>8.2f} '
              f'{float(r["ssim"]):>6.4f} {float(r["lpips"]):>6.4f} {pts:>8,}')
    print(f'{"="*80}')
    print(f'\nResults saved to: {RESULTS_CSV}')


# ── メイン ─────────────────────────────────────────────────

def main():
    print('Barbershop Ablation Study (pruning bug fix 適用済み)')
    print('段階的に要素を追加してフル実装を復元\n')

    init_csv()
    record_reference_results()

    t_start = time.time()
    for i, (method_name, method_info) in enumerate(ABLATION_METHODS.items(), 1):
        run_experiment(method_name, method_info)
        elapsed = (time.time() - t_start) / 60
        print(f'  Progress: {i}/{len(ABLATION_METHODS)} | Elapsed: {elapsed:.0f} min')

    print_summary()


if __name__ == '__main__':
    main()
