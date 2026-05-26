#!/usr/bin/env python3
"""RECAP実験: REgion-Corrected Adaptive Pruning

3つの実験条件:
  recap_spags   : プルーニングなし（オリジナルSPaGS相当）
  recap_opacity : 従来のOpacityプルーニング（ベースライン）
  recap          : 提案手法（緯度帯正規化プルーニング）

オリジナルSPaGSハイパーパラメータを使用:
  DENSIFY_GRAD_THRESHOLD : 0.0002
  DENSIFY_START_ITERATION: 500
"""

import subprocess, yaml, tempfile, os, time
from pathlib import Path

SCENES = [
    'barbershop', 'archiviz-flat', 'bistro_bike', 'bistro_square',
    'classroom', 'fisher-hut', 'lone_monk', 'pavilion_midday_chair',
    'pavilion_midday_pond', 'restroom', 'LOU',
]

ORIG_HP = {
    'DENSIFY_GRAD_THRESHOLD':  0.0002,
    'DENSIFY_START_ITERATION': 500,
}

METHODS = {
    # ── プルーニングなし（オリジナルSPaGS相当）──
    'recap_spags': {
        **ORIG_HP,
        'USE_WS_LOSS':              False,
        'USE_CONTRIBUTION_PRUNING': False,
        'SA_OPACITY_PRUNING':       False,
        'USE_LAT_DENSIFY_CORRECTION': False,
        'EXPERIMENT_TAG':           'recap_spags',
        'WANDB_GROUP':              'recap_spags',
    },
    # ── 従来手法: Opacityプルーニング ──
    'recap_opacity': {
        **ORIG_HP,
        'USE_CONTRIBUTION_PRUNING': True,
        'USE_OPACITY_BASELINE':     True,
        'OPACITY_BASELINE_KEEP_RATIO': 0.690,
        'USE_LATITUDE_PRUNING':     False,
        'USE_REGION_NORMALIZED':    False,
        'SA_OPACITY_PRUNING':       False,
        'USE_LAT_DENSIFY_CORRECTION': False,
        'EXPERIMENT_TAG':           'recap_opacity',
        'WANDB_GROUP':              'recap_opacity',
    },
    # ── 提案手法: 緯度帯正規化プルーニング（RECAP）──
    'recap': {
        **ORIG_HP,
        'USE_CONTRIBUTION_PRUNING': True,
        'USE_LATITUDE_PRUNING':     True,
        'LATITUDE_PRUNING_KEEP_RATIO': 0.690,
        'LATITUDE_BAND_LOW_DEG':    30.0,
        'LATITUDE_BAND_HIGH_DEG':   60.0,
        'USE_OPACITY_BASELINE':     False,
        'USE_REGION_NORMALIZED':    False,
        'SA_OPACITY_PRUNING':       False,
        'USE_LAT_DENSIFY_CORRECTION': False,
        'EXPERIMENT_TAG':           'recap',
        'WANDB_GROUP':              'recap',
    },
}

WANDB_PROJECT = 'RECAP'
RUN_SCENES    = SCENES
RUN_METHODS   = ['recap_spags', 'recap_opacity', 'recap']


def find_completed(scene, method):
    root = Path('output/SPaGS')
    for d in sorted(root.glob(f'{scene}_{method}_*'), reverse=True):
        if (d / 'checkpoints' / 'final.pt').exists():
            return d
    return None


def run(scene, method, skip_completed=True):
    if skip_completed and find_completed(scene, method):
        print(f'  SKIP: {scene}/{method} already done')
        return True

    with open(f'configs/{scene}.yaml') as f:
        config = yaml.safe_load(f)

    cfg = METHODS[method]
    config['TRAINING']['WANDB']['ACTIVATE'] = True
    config['TRAINING']['WANDB']['PROJECT']  = WANDB_PROJECT
    config['TRAINING']['WANDB']['GROUP']    = cfg.get('WANDB_GROUP', method)
    config['TRAINING']['MODEL_NAME']        = f'{scene}_{method}'
    for k, v in cfg.items():
        if k == 'WANDB_GROUP':
            continue
        config['TRAINING'][k] = v

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        tmp = f.name

    print(f'\n{"="*60}\n  {scene} / {method}\n{"="*60}')
    t0 = time.time()
    try:
        subprocess.run(['python', 'scripts/train.py', '-c', tmp], check=True)
        print(f'  Done in {(time.time()-t0)/60:.1f} min')
        return True
    except subprocess.CalledProcessError as e:
        print(f'  FAILED: {e}')
        return False
    finally:
        os.unlink(tmp)


def main():
    queue = [(s, m) for s in RUN_SCENES for m in RUN_METHODS]
    total, done, failed = len(queue), 0, []
    t_start = time.time()
    print(f'RECAP実験: {total} runs ({len(RUN_SCENES)} scenes × {len(RUN_METHODS)} methods)')
    print(f'  WandB project: {WANDB_PROJECT}')
    print(f'  Methods: {RUN_METHODS}')

    for scene, method in queue:
        ok = run(scene, method)
        done += 1
        if not ok:
            failed.append(f'{scene}/{method}')
        elapsed = (time.time() - t_start) / 60
        eta = elapsed / done * (total - done)
        print(f'  Progress: {done}/{total} | Elapsed: {elapsed:.0f} min | ETA: {eta:.0f} min')

    print(f'\nDone. {done-len(failed)}/{total} succeeded.')
    if failed:
        print(f'Failed: {failed}')


if __name__ == '__main__':
    main()
