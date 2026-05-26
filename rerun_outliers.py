#!/usr/bin/env python3
"""bistro_bike/proposed_v0 と bistro_square/proposed_v1 の再実行。"""

import subprocess, yaml, tempfile, os, time
from pathlib import Path

TARGETS = [
    ('bistro_bike',   'proposed_v0'),
    ('bistro_square', 'proposed_v1'),
]

METHODS = {
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

for scene, method in TARGETS:
    config_path = Path(f'configs/{scene}.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config['TRAINING']['WANDB']['ACTIVATE'] = True
    config['TRAINING']['WANDB']['PROJECT']  = 'OmniPrune'
    config['TRAINING']['MODEL_NAME']        = f'{scene}_{method}'
    for key, val in METHODS[method].items():
        config['TRAINING'][key] = val

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        tmp = f.name

    print(f'\n{"="*60}')
    print(f'  Scene : {scene}')
    print(f'  Method: {method}')
    print(f'{"="*60}')

    t0 = time.time()
    try:
        subprocess.run(['python', 'scripts/train.py', '-c', tmp], check=True)
        print(f'  Done in {(time.time()-t0)/60:.1f} min')
    except subprocess.CalledProcessError as e:
        print(f'  FAILED: {e}')
    finally:
        os.unlink(tmp)

print('\nAll done.')
