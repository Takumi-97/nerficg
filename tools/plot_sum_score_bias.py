#!/usr/bin/env python3
"""
SUMベーススコア（opacity × pixel_coverage）の仰角分布を可視化。
pixel_coverage ∝ sigma^2 / (r^2 * cos(theta)) なので 1/cos(theta) バイアスが出る。
チェックポイントのみ使用。GPU/Dataset/Framework不要。
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import math

SCENES = {
    "barbershop":    "output/SPaGS/barbershop_v2_spags_2026-05-22-12-28-44",
    "archiviz-flat": "output/SPaGS/archiviz-flat_v2_spags_2026-05-22-13-36-01",
    "bistro_bike":   "output/SPaGS/bistro_bike_v2_spags_2026-05-22-14-12-41",
    "bistro_square": "output/SPaGS/bistro_square_v2_spags_2026-05-22-18-27-13",
    "classroom":     "output/SPaGS/classroom_v2_spags_2026-05-22-19-18-07",
    "fisher-hut":    "output/SPaGS/fisher-hut_v2_spags_2026-05-26-11-01-32",
    "lone_monk":     "output/SPaGS/lone_monk_v2_spags_2026-05-26-11-23-32",
}

BINS    = np.linspace(-90, 90, 19)
CENTERS = 0.5 * (BINS[:-1] + BINS[1:])


def compute_sum_score(exp_dir):
    pt = Path(exp_dir) / "checkpoints" / "final.pt"
    sd = torch.load(str(pt), map_location="cpu", weights_only=False)["model_state_dict"]

    pos        = sd["gaussians._positions"].numpy()           # [N, 3]
    scales_act = sd["gaussians._scales"].numpy()              # [N, 3]  already activated
    opa        = sd["gaussians._opacities"].numpy().squeeze() # [N]     already sigmoid

    # 仰角（ワールド座標）
    dist_xz  = np.sqrt(pos[:, 0]**2 + pos[:, 2]**2).clip(1e-6)
    r        = np.sqrt((pos**2).sum(axis=1)).clip(1e-6)
    theta    = np.arctan2(pos[:, 1], dist_xz)
    elev_deg = np.degrees(theta)
    cos_th   = np.cos(theta).clip(0.05)

    # Gaussianの有効サイズ（幾何平均）
    sigma = np.cbrt(np.prod(np.abs(scales_act), axis=1)).clip(1e-8)

    # 立体角 ∝ sigma^2 / r^2
    solid_angle = sigma**2 / r**2

    # equirectangularピクセルカバレッジ = solid_angle / cos(theta)
    pixel_coverage = solid_angle / cos_th

    # SUMベーススコア = opacity × pixel_coverage
    score = opa * pixel_coverage

    return elev_deg, score, cos_th


def bin_normalize(elev_deg, scores):
    binned = np.full(len(CENTERS), np.nan)
    for i, (lo, hi) in enumerate(zip(BINS[:-1], BINS[1:])):
        m = (elev_deg >= lo) & (elev_deg < hi)
        if m.sum() >= 5:
            binned[i] = scores[m].mean()
    eq = np.nanmean(binned[np.abs(CENTERS) < 15])
    return binned / eq if eq > 0 else binned


def main():
    all_norm = {}
    for scene_name, exp_dir in SCENES.items():
        pt = Path(exp_dir) / "checkpoints" / "final.pt"
        if not pt.exists():
            print(f"[SKIP] {scene_name}")
            continue
        elev, score, _ = compute_sum_score(exp_dir)
        norm = bin_normalize(elev, score)
        all_norm[scene_name] = norm
        print(f"{scene_name}: done (N={len(elev):,})")

    stacked = np.stack(list(all_norm.values()), axis=0)
    agg     = np.nanmean(stacked, axis=0)
    std     = np.nanstd(stacked, axis=0)

    # 理論曲線
    cos_th  = np.cos(np.radians(CENTERS)).clip(0.05)
    theory  = (1 / cos_th) / np.mean(1 / cos_th[np.abs(CENTERS) < 15])

    # ── プロット ──
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axvspan(-90, -60, alpha=0.10, color='#4477AA', zorder=0)
    ax.axvspan( 60,  90, alpha=0.10, color='#4477AA', zorder=0)

    valid = ~np.isnan(agg)
    ax.fill_between(CENTERS[valid], (agg - std)[valid], (agg + std)[valid],
                    color='#E05A2B', alpha=0.25, label='±1 std (across scenes)')
    ax.plot(CENTERS[valid], agg[valid],
            color='#E05A2B', lw=2.5, marker='o', markersize=4,
            zorder=5, label='Mean score (no correction)')
    ax.plot(CENTERS[valid], theory[valid],
            color='#4477AA', lw=2.0, ls='--', zorder=4,
            label='1/cos(θ) theory')
    ax.axhline(1.0, color='gray', lw=1.2, ls=':', label='Equator reference (=1)')

    ymax = 5.0
    ax.set_ylim(0, ymax)
    ax.text(-75, ymax * 0.95, 'Polar\nzone', ha='center', va='top',
            fontsize=9, color='#4477AA', style='italic')
    ax.text( 75, ymax * 0.95, 'Polar\nzone', ha='center', va='top',
            fontsize=9, color='#4477AA', style='italic')

    ax.set_xlabel('Gaussian elevation angle [deg]', fontsize=12)
    ax.set_ylabel('Normalized importance score\n(equatorial Gaussians = 1.0)', fontsize=11)
    ax.set_title('Score inflation in conventional pruning\n'
                 '(SUM-based proxy: opacity × pixel coverage ∝ 1/cos θ)', fontsize=12)
    ax.set_xlim(-90, 90)
    ax.set_xticks(range(-90, 91, 15))
    ax.legend(fontsize=10, loc='upper center')
    ax.grid(ls='--', alpha=0.35)

    fig.tight_layout()
    out = Path("results/figures/sum_score_bias.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved → {out}")

    print(f"\n{'Lat':>8} | {'Mean':>6} | {'Std':>5} | {'Theory':>7}")
    print("-" * 35)
    for i, lat in enumerate(CENTERS):
        if not np.isnan(agg[i]):
            print(f"{lat:>+8.1f}° | {agg[i]:>6.3f} | {std[i]:>5.3f} | {theory[i]:>7.3f}")


if __name__ == '__main__':
    main()
