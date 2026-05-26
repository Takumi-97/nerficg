#!/usr/bin/env python3
"""camcos実験を全シーンで先に実行し、その後v2の残り実験を続ける。

実行順:
  Phase 1: proposed_camcos × 11 scenes
  Phase 2: v2_spags / v2_baseline / v2_opacity / v2_proposed × 11 scenes（完了済みはスキップ）
"""

import subprocess, yaml, tempfile, os, time
from pathlib import Path

# run_v2.py から設定を流用
from run_v2 import SCENES, METHODS, WANDB_PROJECT, find_completed

PHASE1_METHODS = ['proposed_camcos']
PHASE2_METHODS = ['v2_spags', 'v2_baseline', 'v2_opacity', 'v2_proposed']


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


def run_phase(phase_name, methods, scenes):
    queue = [(s, m) for s in scenes for m in methods]
    total, done, failed = len(queue), 0, []
    t_start = time.time()
    print(f'\n{"#"*60}')
    print(f'# {phase_name}: {total} runs')
    print(f'{"#"*60}')

    for scene, method in queue:
        ok = run(scene, method)
        done += 1
        if not ok:
            failed.append(f'{scene}/{method}')
        elapsed = (time.time() - t_start) / 60
        eta = elapsed / done * (total - done)
        print(f'  Progress: {done}/{total} | Elapsed: {elapsed:.0f} min | ETA: {eta:.0f} min')

    print(f'\n{phase_name} done. {done-len(failed)}/{total} succeeded.')
    if failed:
        print(f'Failed: {failed}')
    return failed


if __name__ == '__main__':
    all_failed = []
    all_failed += run_phase('Phase 1: proposed_camcos', PHASE1_METHODS, SCENES)
    all_failed += run_phase('Phase 2: v2 experiments',  PHASE2_METHODS, SCENES)

    print(f'\n{"="*60}')
    print(f'All done. Failed: {all_failed if all_failed else "none"}')
