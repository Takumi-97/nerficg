#!/usr/bin/env python3
"""v2_baselineの異常値（barbershop: 23.61, archiviz-flat: 23.93）を強制再実行。"""

import subprocess, yaml, tempfile, os, time
from run_v2 import METHODS, WANDB_PROJECT

FORCE_RERUN = [
    ('barbershop',   'v2_baseline'),
    ('archiviz-flat','v2_baseline'),
]


def run(scene, method):
    with open(f'configs/{scene}.yaml') as f:
        config = yaml.safe_load(f)

    config['TRAINING']['WANDB']['ACTIVATE'] = True
    config['TRAINING']['WANDB']['PROJECT']  = WANDB_PROJECT
    config['TRAINING']['MODEL_NAME']        = f'{scene}_{method}'
    for k, v in METHODS[method].items():
        config['TRAINING'][k] = v

    print(f'\n{"="*60}\n  {scene} / {method}  [FORCE RERUN]\n{"="*60}')
    t0 = time.time()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        tmp = f.name

    try:
        subprocess.run(['python', 'scripts/train.py', '-c', tmp], check=True)
        print(f'  Done in {(time.time()-t0)/60:.1f} min')
        return True
    except subprocess.CalledProcessError as e:
        print(f'  FAILED: {e}')
        return False
    finally:
        os.unlink(tmp)


if __name__ == '__main__':
    failed = []
    for scene, method in FORCE_RERUN:
        if not run(scene, method):
            failed.append(f'{scene}/{method}')
    print(f'\nDone. Failed: {failed if failed else "none"}')
