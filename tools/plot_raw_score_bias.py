#!/usr/bin/env python3
"""
補正なし（従来手法）のGaussian重要度スコアの仰角分布を可視化する。
極付近のGaussianが過剰評価されることを示す動機付け図。
"""
import os
os.environ["WANDB_MODE"] = "disabled"
import sys
import math
import random
import shutil
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

SCENES = {
    "barbershop":    "output/SPaGS/barbershop_v2_spags_2026-05-22-12-28-44",
    "archiviz-flat": "output/SPaGS/archiviz-flat_v2_spags_2026-05-22-13-36-01",
    "bistro_bike":   "output/SPaGS/bistro_bike_v2_spags_2026-05-22-14-12-41",
    "bistro_square": "output/SPaGS/bistro_square_v2_spags_2026-05-22-18-27-13",
    "classroom":     "output/SPaGS/classroom_v2_spags_2026-05-22-19-18-07",
    "fisher-hut":    "output/SPaGS/fisher-hut_v2_spags_2026-05-26-11-01-32",
    "lone_monk":     "output/SPaGS/lone_monk_v2_spags_2026-05-26-11-23-32",
}

N_FRAMES = 50
BINS    = np.linspace(-90, 90, 19)   # 10° ビン
CENTERS = 0.5 * (BINS[:-1] + BINS[1:])


def compute_raw_scores(exp_dir, n_frames):
    import utils
    with utils.discoverSourcePath():
        import Framework
        from Implementations import Methods as MI, Datasets as DI

    Framework.setup(config_path=f"{exp_dir}/training_config.yaml", require_custom_config=True)
    try:
        spurious = Path(Framework.config.OUTPUT.PATH)
        if spurious.exists() and spurious != Path(exp_dir):
            shutil.rmtree(spurious, ignore_errors=True)
    except Exception:
        pass

    model_instance = MI.getTrainingInstance(method=Framework.config.GLOBAL.METHOD_TYPE, checkpoint=None)
    state = torch.load(f"{exp_dir}/checkpoints/final.pt", map_location="cuda", weights_only=False)["model_state_dict"]
    g = model_instance.model.gaussians
    for key in ["_positions", "_scales", "_rotations", "_opacities", "_sh_0", "_sh_rest"]:
        full = f"gaussians.{key}"
        if full in state:
            setattr(g, key, torch.nn.Parameter(state[full].cuda()))

    dataset = DI.getDataset(dataset_type=Framework.config.GLOBAL.DATASET_TYPE,
                            path=Framework.config.DATASET.PATH)
    dataset.setMode("train")
    renderer = model_instance.renderer

    positions = g.get_positions.detach()
    scales    = g.get_scales.detach()
    rotations = g.get_rotations.detach()
    opacities = g.get_opacities.detach()
    n = positions.shape[0]

    dist_xz  = torch.norm(positions[:, [0, 2]], dim=1).clamp(1e-6)
    theta_w  = torch.atan2(positions[:, 1], dist_xz)
    elev_deg = (theta_w * 180 / math.pi).cpu().numpy()

    random.seed(42)
    all_props = list(dataset.data[dataset.mode])
    sampled   = random.sample(all_props, min(n_frames, len(all_props)))

    scores_sum = torch.zeros(n, device="cuda")
    for props in sampled:
        dataset.camera.setProperties(props)
        fw = torch.zeros(n, device="cuda")
        renderer.rasterizer.update_max_weights(
            max_weights=fw, positions=positions, scales=scales,
            rotations=rotations, opacities=opacities,
            camera=dataset.camera, mode=renderer.BLEND_MODE, K=renderer.K,
            active_sh_bases=0, scale_modifier=1.0, weight_threshold=0.01,
        )
        scores_sum += fw

    Framework.teardown()
    avg_score = (scores_sum / len(sampled)).cpu().numpy()
    return elev_deg, avg_score


def bin_and_normalize(elev_deg, scores):
    binned = np.full(len(CENTERS), np.nan)
    for i, (lo, hi) in enumerate(zip(BINS[:-1], BINS[1:])):
        m = (elev_deg >= lo) & (elev_deg < hi) & (scores > 0)
        if m.sum() >= 3:
            binned[i] = scores[m].mean()
    eq = np.nanmean(binned[np.abs(CENTERS) < 15])
    return binned / eq if (eq and eq > 0) else binned


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    all_norm = {}
    for scene_name, exp_dir in SCENES.items():
        if not Path(f"{exp_dir}/checkpoints/final.pt").exists():
            print(f"[SKIP] {scene_name}")
            continue
        print(f"Processing {scene_name}...")
        elev, scores = compute_raw_scores(exp_dir, N_FRAMES)
        norm = bin_and_normalize(elev, scores)
        all_norm[scene_name] = norm
        print(f"  done. non-nan bins: {(~np.isnan(norm)).sum()}")

    if not all_norm:
        print("No data.")
        return

    agg = np.nanmean(list(all_norm.values()), axis=0)

    # ── プロット ──
    fig, ax = plt.subplots(figsize=(8, 5))

    # 極ゾーンのハイライト
    ax.axvspan(-90, -60, alpha=0.10, color='#4477AA', zorder=0)
    ax.axvspan( 60,  90, alpha=0.10, color='#4477AA', zorder=0)

    # mean ± std
    stacked = np.stack(list(all_norm.values()), axis=0)   # [n_scenes, n_bins]
    std = np.nanstd(stacked, axis=0)

    valid_agg = ~np.isnan(agg)
    ax.fill_between(CENTERS[valid_agg],
                    (agg - std)[valid_agg], (agg + std)[valid_agg],
                    color='#E05A2B', alpha=0.25, label='±1 std (across scenes)')
    ax.plot(CENTERS[valid_agg], agg[valid_agg],
            color='#E05A2B', lw=2.5, marker='o', markersize=4,
            zorder=5, label='Mean (all scenes)')

    # 赤道基準ライン
    ax.axhline(1.0, color='gray', lw=1.5, ls='--', zorder=4, label='Equator reference (=1)')

    ymax = 4.0
    ax.set_ylim(0, ymax)

    # 極ゾーンラベル
    ax.text(-75, ymax * 0.95, 'Polar\nzone',
            ha='center', va='top', fontsize=9, color='#4477AA', style='italic')
    ax.text( 75, ymax * 0.95, 'Polar\nzone',
            ha='center', va='top', fontsize=9, color='#4477AA', style='italic')

    ax.set_xlabel('Gaussian elevation angle [deg]', fontsize=12)
    ax.set_ylabel('Normalized importance score\n(equatorial Gaussians = 1.0)', fontsize=11)
    ax.set_title('Conventional pruning scores by elevation\n'
                 '(without correction: non-equatorial Gaussians over-scored)', fontsize=12)
    ax.set_xlim(-90, 90)
    ax.set_xticks(range(-90, 91, 15))
    ax.legend(fontsize=10, loc='upper center')
    ax.grid(ls='--', alpha=0.35)

    fig.tight_layout()
    out = Path("results/figures/raw_score_bias.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved → {out}")

    # 数値サマリ
    print(f"\n{'Lat':>8} | {'Mean score':>10} | {'Max across scenes':>18}")
    print("-" * 45)
    for i, lat in enumerate(CENTERS):
        vals = [v[i] for v in all_norm.values() if not np.isnan(v[i])]
        if vals:
            print(f"{lat:>+8.1f}° | {np.mean(vals):>10.3f} | {np.max(vals):>18.3f}")


if __name__ == '__main__':
    main()
