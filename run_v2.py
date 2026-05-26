#!/usr/bin/env python3
"""実験V2：オリジナルSPaGSハイパーパラメータ + cam_spherical アブレーション

変更点（オリジナルSPaGS寄せ）:
  DENSIFY_GRAD_THRESHOLD : 0.00005 → 0.0002
  DENSIFY_START_ITERATION: 200    → 500

追加実験:
  proposed_camcos: cos補正をフレームループ内でカメラ相対方向から計算
"""

import subprocess, yaml, tempfile, os, time
from pathlib import Path

SCENES = [
    'barbershop', 'archiviz-flat', 'bistro_bike', 'bistro_square',
    'classroom', 'fisher-hut', 'lone_monk', 'pavilion_midday_chair',
    'pavilion_midday_pond', 'restroom', 'LOU',
]

# オリジナルHP共通上書き
ORIG_HP = {
    'DENSIFY_GRAD_THRESHOLD':  0.0002,
    'DENSIFY_START_ITERATION': 500,
}

METHODS = {
    # ── オリジナルHP版ベースライン ──
    'v2_spags': {
        **ORIG_HP,
        'USE_WS_LOSS':              False,
        'USE_CONTRIBUTION_PRUNING': False,
        'USE_OPACITY_BASELINE':     False,
        'USE_REGION_NORMALIZED':    False,
        'SA_OPACITY_PRUNING':       False,
        'USE_LAT_DENSIFY_CORRECTION': False,
        'EXPERIMENT_TAG':           'v2_spags',
    },
    'v2_baseline': {
        **ORIG_HP,
        'USE_CONTRIBUTION_PRUNING': False,
        'USE_OPACITY_BASELINE':     False,
        'USE_REGION_NORMALIZED':    False,
        'USE_ERROR_WEIGHT':         False,
        'SA_OPACITY_PRUNING':       False,
        'USE_LAT_DENSIFY_CORRECTION': False,
        'EXPERIMENT_TAG':           'v2_baseline',
    },
    'v2_opacity': {
        **ORIG_HP,
        'USE_CONTRIBUTION_PRUNING': True,
        'USE_OPACITY_BASELINE':     True,
        'USE_REGION_NORMALIZED':    False,
        'USE_ERROR_WEIGHT':         False,
        'SA_OPACITY_PRUNING':       False,
        'EXPERIMENT_TAG':           'v2_opacity',
    },
    'v2_proposed': {
        **ORIG_HP,
        'USE_CONTRIBUTION_PRUNING':        True,
        'USE_OPACITY_BASELINE':            False,
        'USE_REGION_NORMALIZED':           True,
        'CONTRIBUTION_PRUNING_USE_VOLUME': False,
        'USE_ERROR_WEIGHT':                False,
        'SA_OPACITY_PRUNING':              True,
        'USE_LAT_DENSIFY_CORRECTION':      False,
        'EXPERIMENT_TAG':                  'v2_proposed',
    },
    # ── cam_spherical アブレーション（現行HPで実施）──
    'proposed_camcos': {
        'USE_CONTRIBUTION_PRUNING':           True,
        'USE_OPACITY_BASELINE':               False,
        'USE_REGION_NORMALIZED':              True,
        'CONTRIBUTION_PRUNING_USE_VOLUME':    False,
        'CONTRIBUTION_PRUNING_USE_SPHERICAL': False,   # グローバルcos OFF
        'CONTRIBUTION_PRUNING_USE_CAM_SPHERICAL': True, # フレーム内カメラ相対cos ON
        'USE_ERROR_WEIGHT':                   False,
        'SA_OPACITY_PRUNING':                 True,
        'USE_LAT_DENSIFY_CORRECTION':         False,
        'EXPERIMENT_TAG':                     'proposed_camcos',
    },
}

WANDB_PROJECT = 'OmniPrune'
RUN_SCENES  = SCENES
RUN_METHODS = ['v2_spags', 'v2_baseline', 'v2_opacity', 'v2_proposed']


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

    config['TRAINING']['WANDB']['ACTIVATE'] = True
    config['TRAINING']['WANDB']['PROJECT']  = WANDB_PROJECT
    config['TRAINING']['MODEL_NAME']        = f'{scene}_{method}'
    for k, v in METHODS[method].items():
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
    print(f'V2実験: {total} runs ({len(RUN_SCENES)} scenes × {len(RUN_METHODS)} methods)')

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
