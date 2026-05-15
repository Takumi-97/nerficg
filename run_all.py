#!/usr/bin/env python3
"""OmniPrune: 全実験実行スクリプト（9シーン × 3手法 = 27 runs）"""

import subprocess
import yaml
import tempfile
import os
import time
from pathlib import Path

# ── 実験設定 ──────────────────────────────────────────────
SCENES = [
    'barbershop',
    'archiviz-flat',
    'bistro_bike',
    'bistro_square',
    'classroom',
    'fisher-hut',
    'lone_monk',
    'pavilion_midday_chair',
    'pavilion_midday_pond',
    'restroom',
]

METHODS = {
    'baseline': {
        'USE_CONTRIBUTION_PRUNING': False,
        'USE_OPACITY_BASELINE':     False,
        'USE_OPACITY_SCORE':        False,
        'USE_UNIFORM_CONTRIBUTION': False,
        'USE_REGION_NORMALIZED':    False,
        'USE_ERROR_WEIGHT':         False,
        'EXPERIMENT_TAG':           'baseline',
    },
    'opacity': {
        'USE_CONTRIBUTION_PRUNING': True,
        'USE_OPACITY_BASELINE':     True,
        'USE_OPACITY_SCORE':        False,
        'USE_UNIFORM_CONTRIBUTION': False,
        'USE_REGION_NORMALIZED':    False,
        'USE_ERROR_WEIGHT':         False,
        'EXPERIMENT_TAG':           'opacity',
    },
    'proposed': {
        'USE_CONTRIBUTION_PRUNING': True,
        'USE_OPACITY_BASELINE':     False,
        'USE_OPACITY_SCORE':        False,
        'USE_UNIFORM_CONTRIBUTION': False,
        'USE_REGION_NORMALIZED':    True,
        'USE_ERROR_WEIGHT':         True,
        'ERROR_WEIGHT_LAMBDA':      2.0,
        'EXPERIMENT_TAG':           'proposed',
    },
}

WANDB_ENTITY  = None  # nullにしてwandbがAPIキーから自動判定
WANDB_PROJECT = 'OmniPrune'

# ── 実行制御 ──────────────────────────────────────────────
# 特定のシーン/手法だけ実行したい場合はここを変更
RUN_SCENES  = [
    'bistro_bike',
    'bistro_square',
    'classroom',
    'fisher-hut',
    'lone_monk',
    'pavilion_midday_chair',
    'pavilion_midday_pond',
    'restroom',
]
RUN_METHODS = ['baseline', 'opacity', 'proposed']


def load_base_config(scene: str) -> dict:
    config_path = Path(f'configs/{scene}.yaml')
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_completed_run(scene: str, method: str) -> Path | None:
    """final.pt が存在する最新の出力ディレクトリを返す。なければ None。"""
    output_root = Path('output/SPaGS')
    prefix = f'{scene}_{method}_'
    candidates = sorted(output_root.glob(f'{prefix}*'), reverse=True)
    for d in candidates:
        if (d / 'checkpoints' / 'final.pt').exists():
            return d
    return None


def run_experiment(scene: str, method: str, skip_completed: bool = True) -> bool:
    if skip_completed:
        completed = find_completed_run(scene, method)
        if completed:
            print(f'\n{"="*60}')
            print(f'  Scene : {scene}')
            print(f'  Method: {method}')
            print(f'  SKIP  : already completed → {completed.name}')
            print(f'{"="*60}')
            return True

    config = load_base_config(scene)

    # wandb有効化
    config['TRAINING']['WANDB']['ACTIVATE'] = True
    config['TRAINING']['WANDB']['PROJECT']  = WANDB_PROJECT
    if WANDB_ENTITY is not None:
        config['TRAINING']['WANDB']['ENTITY'] = WANDB_ENTITY

    # run名: {scene}_{method}
    config['TRAINING']['MODEL_NAME'] = f'{scene}_{method}'

    # 手法フラグを上書き
    for key, val in METHODS[method].items():
        config['TRAINING'][key] = val

    # 一時configに書き出し
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        temp_path = f.name

    print(f'\n{"="*60}')
    print(f'  Scene : {scene}')
    print(f'  Method: {method}')
    print(f'{"="*60}')

    t0 = time.time()
    try:
        subprocess.run(
            ['python', 'scripts/train.py', '-c', temp_path],
            check=True
        )
        elapsed = (time.time() - t0) / 60
        print(f'  ✓ Done in {elapsed:.1f} min')
        return True
    except subprocess.CalledProcessError as e:
        print(f'  ✗ FAILED: {e}')
        return False
    finally:
        os.unlink(temp_path)


def main():
    total   = len(RUN_SCENES) * len(RUN_METHODS)
    done    = 0
    failed  = []
    t_start = time.time()

    print(f'OmniPrune: {total} experiments ({len(RUN_SCENES)} scenes × {len(RUN_METHODS)} methods)')
    print(f'wandb project: {WANDB_PROJECT} / entity: {WANDB_ENTITY}\n')

    for scene in RUN_SCENES:
        for method in RUN_METHODS:
            success = run_experiment(scene, method)
            done += 1
            if not success:
                failed.append(f'{scene}/{method}')
            elapsed_total = (time.time() - t_start) / 60
            remaining = total - done
            eta = (elapsed_total / done * remaining) if done > 0 else 0
            print(f'  Progress: {done}/{total} | Elapsed: {elapsed_total:.0f} min | ETA: {eta:.0f} min')

    print(f'\n{"="*60}')
    print(f'All done. {done - len(failed)}/{total} succeeded.')
    if failed:
        print(f'Failed: {failed}')


if __name__ == '__main__':
    main()
