#!/usr/bin/env python3
"""定性比較図（レンダリング結果 + エラーマップ）を生成。

指定シーン・フレームに対して、複数手法の比較画像を
[GT | method1 | method2 | ...] + エラーマップの形式で出力する。

使用例:
  python tools/plot_qualitative.py
  python tools/plot_qualitative.py --scenes barbershop lone_monk --frames 0 12
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image


OUTPUT_ROOT = Path('output/SPaGS')

# 比較対象メソッド: (method_tag, 表示ラベル)
COMPARE_METHODS = [
    ('spags',       'SPaGS'),
    ('baseline',    'Baseline'),
    ('proposed_v1', 'Proposed'),
]

# シーンごとに実験ディレクトリを特定（最新の完了済みを使う）
def find_exp_dir(scene: str, method_tag: str) -> Path | None:
    candidates = sorted(
        OUTPUT_ROOT.glob(f'{scene}_{method_tag}_*'), reverse=True
    )
    for d in candidates:
        if (d / 'checkpoints' / 'final.pt').exists():
            return d
        if (d / 'test_30000' / 'rgb').exists():
            return d
    return None


def load_image(path: Path) -> np.ndarray | None:
    try:
        img = Image.open(path).convert('RGB')
        return np.array(img, dtype=np.float32) / 255.0
    except Exception:
        return None


def compute_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """絶対誤差をグレースケールで返す（0〜1クリップ）。"""
    return np.abs(pred - gt).mean(axis=2)


def plot_qualitative_scene(scene: str, frame_ids: list[int], output_dir: Path):
    methods = []
    for tag, label in COMPARE_METHODS:
        d = find_exp_dir(scene, tag)
        if d is None:
            print(f'  SKIP: {scene}/{tag} not found')
            continue
        rgb_dir = d / 'test_30000' / 'rgb'
        gt_dir  = d / 'test_30000' / 'rgb_gt'
        if not rgb_dir.exists():
            print(f'  SKIP: {scene}/{tag} no test_30000/rgb')
            continue
        methods.append({'tag': tag, 'label': label, 'rgb_dir': rgb_dir, 'gt_dir': gt_dir})

    if not methods:
        print(f'  No methods found for {scene}')
        return

    for fid in frame_ids:
        fname = f'{fid:05d}.png'

        # GT を最初のメソッドから取得
        gt_path = methods[0]['gt_dir'] / fname
        gt = load_image(gt_path)
        if gt is None:
            print(f'  SKIP: GT not found ({gt_path})')
            continue

        # 各メソッドの画像とエラーマップを収集
        preds   = []
        errors  = []
        labels  = []
        for m in methods:
            pred = load_image(m['rgb_dir'] / fname)
            if pred is None:
                continue
            preds.append(pred)
            errors.append(compute_error(pred, gt))
            labels.append(m['label'])

        if not preds:
            continue

        n_methods = len(preds)
        # レイアウト: 上段=GT + 各メソッド, 下段=GT（空白）+ エラーマップ
        fig = plt.figure(figsize=((n_methods + 1) * 3.2, 5.5))
        gs = gridspec.GridSpec(2, n_methods + 1, figure=fig,
                               hspace=0.04, wspace=0.03)

        # ── 上段 ──
        ax_gt = fig.add_subplot(gs[0, 0])
        ax_gt.imshow(gt)
        ax_gt.set_title('Ground Truth', fontsize=9, pad=3)
        ax_gt.axis('off')

        for i, (pred, label) in enumerate(zip(preds, labels)):
            ax = fig.add_subplot(gs[0, i + 1])
            ax.imshow(pred)
            ax.set_title(label, fontsize=9, pad=3)
            ax.axis('off')

        # ── 下段（エラーマップ）──
        ax_blank = fig.add_subplot(gs[1, 0])
        ax_blank.axis('off')

        vmax = max(e.max() for e in errors) * 0.6  # 飽和を抑える

        for i, (err, label) in enumerate(zip(errors, labels)):
            ax = fig.add_subplot(gs[1, i + 1])
            im = ax.imshow(err, cmap='hot', vmin=0, vmax=vmax)
            ax.set_title(f'Error: {label}', fontsize=8, pad=2)
            ax.axis('off')

        # カラーバー
        cbar_ax = fig.add_axes([0.92, 0.08, 0.012, 0.38])
        fig.colorbar(im, cax=cbar_ax)
        cbar_ax.set_ylabel('L1 Error', fontsize=7)

        out = output_dir / f'qualitative_{scene}_f{fid:05d}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved → {out}')


def main(scenes: list[str], frame_ids: list[int], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for scene in scenes:
        print(f'\n--- {scene} ---')
        plot_qualitative_scene(scene, frame_ids, output_dir)


if __name__ == '__main__':
    default_scenes = ['barbershop', 'lone_monk', 'classroom']
    default_frames = [0, 6, 12]

    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes', nargs='+', default=default_scenes)
    parser.add_argument('--frames', nargs='+', type=int, default=default_frames)
    parser.add_argument('--outdir', default='results/figures', type=Path)
    args = parser.parse_args()
    main(args.scenes, args.frames, args.outdir)
