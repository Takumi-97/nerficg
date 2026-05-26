#!/usr/bin/env python3
"""未完了 + 異常値実験をまとめて再実行するスクリプト。

実行対象:
  通常スキップ（find_completed）:
    lone_monk, pavilion_midday_chair, pavilion_midday_pond, restroom, LOU の v2_*

  強制再実行（異常値のため）:
    fisher-hut: v2_spags(8.94dB), v2_opacity(8.91dB), v2_proposed(8.55dB)
    archiviz-flat: v2_opacity(10.39dB), v2_proposed(16.69dB)
"""

import subprocess, yaml, tempfile, os, time
from pathlib import Path
from run_v2 import METHODS, WANDB_PROJECT, find_completed

SCENES = [
    'LOU',
]

V2_METHODS = ['v2_spags', 'v2_baseline', 'v2_opacity', 'v2_proposed']

FORCE_RERUN = set()


def run(scene, method, force=False):
    if not force and find_completed(scene, method):
        print(f'  SKIP: {scene}/{method} already done')
        return True

    with open(f'configs/{scene}.yaml') as f:
        config = yaml.safe_load(f)

    config['TRAINING']['WANDB']['ACTIVATE'] = True
    config['TRAINING']['WANDB']['PROJECT']  = WANDB_PROJECT
    config['TRAINING']['MODEL_NAME']        = f'{scene}_{method}'
    for k, v in METHODS[method].items():
        config['TRAINING'][k] = v

    tag = '[FORCE RERUN]' if force else ''
    print(f'\n{"="*60}\n  {scene} / {method} {tag}\n{"="*60}')
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


def main():
    queue = [(s, m, (s, m) in FORCE_RERUN) for s in SCENES for m in V2_METHODS]
    total = len(queue)
    done, failed = 0, []
    t_start = time.time()

    print(f'残り + 異常値再実行: 最大{total}件（完了済みはスキップ）')

    for scene, method, force in queue:
        ok = run(scene, method, force=force)
        done += 1
        if not ok:
            failed.append(f'{scene}/{method}')
        elapsed = (time.time() - t_start) / 60
        eta = elapsed / done * (total - done)
        print(f'  Progress: {done}/{total} | Elapsed: {elapsed:.0f} min | ETA: {eta:.0f} min')

    print(f'\nDone. Failed: {failed if failed else "none"}')


if __name__ == '__main__':
    main()
