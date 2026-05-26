#!/usr/bin/env python3
"""SPaGS比較実験: 原著SPaGS（standard L1）ベースの3手法
  spags          : 原著SPaGS（standard L1, pruningなし）
  spags_opacity  : SPaGS + opacity pruning（standard L1）
  spags_proposed : SPaGS + proposed_v1 pruning（standard L1）

wandbでの見分け方:
  現行実験(WS-L1ベース): {scene}_baseline / opacity / proposed_v0 / proposed_v1
  本スクリプト(std L1ベース): {scene}_spags / spags_opacity / spags_proposed
"""

import subprocess
import yaml
import tempfile
import os
import time
from pathlib import Path

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

# USE_WS_LOSS=False で原著SPaGSの損失（standard L1 + DSSIM）を使用
METHODS = {
    # 原著SPaGS: standard L1, pruningなし
    'spags': {
        'USE_WS_LOSS':              False,
        'USE_CONTRIBUTION_PRUNING': False,
        'USE_OPACITY_BASELINE':     False,
        'USE_REGION_NORMALIZED':    False,
        'SA_OPACITY_PRUNING':       False,
        'USE_LAT_DENSIFY_CORRECTION': False,
        'EXPERIMENT_TAG':           'spags',
    },
    # SPaGS + opacity pruning: standard L1, opacity-based explicit pruning
    'spags_opacity': {
        'USE_WS_LOSS':              False,
        'USE_CONTRIBUTION_PRUNING': True,
        'USE_OPACITY_BASELINE':     True,
        'USE_REGION_NORMALIZED':    False,
        'SA_OPACITY_PRUNING':       False,
        'USE_LAT_DENSIFY_CORRECTION': False,
        'EXPERIMENT_TAG':           'spags_opacity',
    },
    # SPaGS + proposed pruning: standard L1, proposed_v1 pruning
    'spags_proposed': {
        'USE_WS_LOSS':              False,
        'USE_CONTRIBUTION_PRUNING': True,
        'USE_OPACITY_BASELINE':     False,
        'USE_REGION_NORMALIZED':    True,
        'CONTRIBUTION_PRUNING_USE_VOLUME': False,
        'USE_ERROR_WEIGHT':         False,
        'SA_OPACITY_PRUNING':       True,
        'USE_LAT_DENSIFY_CORRECTION': False,
        'EXPERIMENT_TAG':           'spags_proposed',
    },
}

WANDB_PROJECT = 'OmniPrune'
WANDB_ENTITY  = None

RUN_SCENES  = SCENES
RUN_METHODS = ['spags', 'spags_opacity', 'spags_proposed']


def load_base_config(scene: str) -> dict:
    with open(f'configs/{scene}.yaml') as f:
        return yaml.safe_load(f)


def find_completed_run(scene: str, method: str) -> Path | None:
    output_root = Path('output/SPaGS')
    candidates = sorted(output_root.glob(f'{scene}_{method}_*'), reverse=True)
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

    config['TRAINING']['WANDB']['ACTIVATE'] = True
    config['TRAINING']['WANDB']['PROJECT']  = WANDB_PROJECT
    if WANDB_ENTITY is not None:
        config['TRAINING']['WANDB']['ENTITY'] = WANDB_ENTITY

    config['TRAINING']['MODEL_NAME'] = f'{scene}_{method}'

    for key, val in METHODS[method].items():
        config['TRAINING'][key] = val

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
    queue = [(s, m, True) for s in RUN_SCENES for m in RUN_METHODS]
    total = len(queue)
    done = 0
    failed = []
    t_start = time.time()

    print(f'SPaGS比較実験: {total} runs ({len(RUN_SCENES)} scenes × {len(RUN_METHODS)} methods)')
    print(f'  spags          : 原著SPaGS (standard L1, no pruning)')
    print(f'  spags_opacity  : SPaGS + opacity pruning (standard L1)')
    print(f'  spags_proposed : SPaGS + proposed_v1 pruning (standard L1)')
    print(f'wandb project: {WANDB_PROJECT}\n')

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
    print(f'Done. {done - len(failed)}/{total} succeeded.')
    if failed:
        print(f'Failed: {failed}')


if __name__ == '__main__':
    main()
