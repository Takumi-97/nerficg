#!/usr/bin/env python3
"""OmniPrune: 全実験実行スクリプト（11シーン × 4手法 = 44 runs）"""

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
    'LOU',
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
    # proposed_v0: 仰角補正 + region_normalized のみ（コア提案手法）
    'proposed_v0': {
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
    # proposed_v1: +SA_OPACITY_PRUNING（ablationで最高性能）
    'proposed_v1': {
        'USE_CONTRIBUTION_PRUNING':        True,
        'USE_OPACITY_BASELINE':            False,
        'USE_OPACITY_SCORE':               False,
        'USE_UNIFORM_CONTRIBUTION':        False,
        'USE_REGION_NORMALIZED':           True,
        'CONTRIBUTION_PRUNING_USE_VOLUME': False,
        'USE_ERROR_WEIGHT':                False,
        'SA_OPACITY_PRUNING':              True,
        'USE_LAT_DENSIFY_CORRECTION':      False,
        'EXPERIMENT_TAG':                  'proposed_v1',
    },
}

WANDB_ENTITY  = None  # nullにしてwandbがAPIキーから自動判定
WANDB_PROJECT = 'OmniPrune'

# ── 実行制御 ──────────────────────────────────────────────
# 全11シーン × 3手法を実行
RUN_SCENES  = [
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
    'LOU',
]
RUN_METHODS = ['baseline', 'opacity', 'proposed_v0', 'proposed_v1']

# 完了済みはスキップ（barbershop_baseline / opacity は今日実施済み）
FORCE_RERUN = []


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
    # 完了済みはスキップ、未完了のみ実行
    queue = [(s, m, True) for s in RUN_SCENES for m in RUN_METHODS]

    total   = len(queue)
    done    = 0
    failed  = []
    t_start = time.time()

    print(f'OmniPrune: {total} experiments ({len(RUN_SCENES)} scenes × {len(RUN_METHODS)} methods)')
    print(f'wandb project: {WANDB_PROJECT} / entity: {WANDB_ENTITY}\n')

    for scene, method, skip in queue:
        success = run_experiment(scene, method, skip_completed=skip)
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
