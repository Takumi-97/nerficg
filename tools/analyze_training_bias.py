#!/usr/bin/env python3
"""
analyze_training_bias.py

学習バイアスの実証実験：
  v2_spags   (標準L1 = 全ピクセル均等) vs
  v2_baseline (WS-L1 = cos(θ)重み付き、極ピクセルを減らす)

仮説：
  equirectangularでは極付近のピクセル密度が 1/cos(θ) 倍
  → 標準L1学習では極のガウシアンが過剰に大きいopacityを学習する
  → WS-L1では補正される

結果：opacity緯度分布の差が仮説を支持するか確認する。
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("results/training_bias")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAT_BINS    = np.linspace(-90, 90, 37)   # 5°刻み
LAT_CENTERS = 0.5 * (LAT_BINS[:-1] + LAT_BINS[1:])

# 完了済み実験パス
SCENE_PATHS = {
    "barbershop": {
        "v2_spags":    "output/SPaGS/barbershop_v2_spags_2026-05-22-12-28-44",
        "v2_baseline": "output/SPaGS/barbershop_v2_baseline_2026-05-22-12-52-20",
    },
    "archiviz-flat": {
        "v2_spags":    "output/SPaGS/archiviz-flat_v2_spags_2026-05-22-13-36-01",
        "v2_baseline": "output/SPaGS/archiviz-flat_v2_baseline_2026-05-22-13-46-12",
    },
    "bistro_bike": {
        "v2_spags":    "output/SPaGS/bistro_bike_v2_spags_2026-05-22-14-12-41",
        "v2_baseline": "output/SPaGS/bistro_bike_v2_baseline_2026-05-22-14-26-25",
    },
    "bistro_square": {
        "v2_spags":    "output/SPaGS/bistro_square_v2_spags_2026-05-22-18-27-13",
        "v2_baseline": "output/SPaGS/bistro_square_v2_baseline_2026-05-22-18-41-30",
    },
    "classroom": {
        "v2_spags":    "output/SPaGS/classroom_v2_spags_2026-05-22-19-18-07",
        "v2_baseline": "output/SPaGS/classroom_v2_baseline_2026-05-22-19-29-46",
    },
    "fisher-hut": {
        "v2_spags":    "output/SPaGS/fisher-hut_v2_spags_2026-05-26-11-01-32",
        "v2_baseline": "output/SPaGS/fisher-hut_v2_baseline_2026-05-22-20-09-24",
    },
    "lone_monk": {
        "v2_spags":    "output/SPaGS/lone_monk_v2_spags_2026-05-26-11-23-32",
        "v2_baseline": "output/SPaGS/lone_monk_v2_baseline_2026-05-26-11-37-26",
    },
}

COLORS = {
    "v2_spags":    "#E05A2B",   # オレンジ（補正なし）
    "v2_baseline": "#4477AA",   # 青（WS-L1補正あり）
}
LABELS = {
    "v2_spags":    "Standard L1 (uniform pixel weighting)",
    "v2_baseline": "WS-L1 (cos(θ)-weighted pixels)",
}


def load_pt(run_dir: str):
    pt_path = Path(run_dir) / "checkpoints" / "final.pt"
    if not pt_path.exists():
        return None
    pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = pt["model_state_dict"]
    pos = sd["gaussians._positions"].numpy()
    opa = sd["gaussians._opacities"].numpy().squeeze()
    return pos, opa


def elevation(pos):
    dist_xz = np.sqrt(pos[:, 0]**2 + pos[:, 2]**2).clip(min=1e-6)
    return np.degrees(np.arctan2(pos[:, 1], dist_xz))


def bin_mean(values, elev):
    result = np.full(len(LAT_CENTERS), np.nan)
    for i, (lo, hi) in enumerate(zip(LAT_BINS[:-1], LAT_BINS[1:])):
        m = (elev >= lo) & (elev < hi)
        if m.sum() >= 3:   # 3個未満は信頼性低いのでnan
            result[i] = values[m].mean()
    return result


def bin_count(elev):
    counts = np.zeros(len(LAT_CENTERS))
    for i, (lo, hi) in enumerate(zip(LAT_BINS[:-1], LAT_BINS[1:])):
        counts[i] = ((elev >= lo) & (elev < hi)).sum()
    return counts


def eq_normalize(arr):
    """赤道帯（|lat|<15°）の平均を1に正規化"""
    eq_mask = np.abs(LAT_CENTERS) < 15
    denom = np.nanmean(arr[eq_mask])
    if denom > 0:
        return arr / denom
    return arr


def analyze_single(scene_name, paths, ax_opa, ax_norm):
    data = {}
    for method, path in paths.items():
        result = load_pt(path)
        if result is None:
            print(f"  [SKIP] {scene_name}/{method}: no final.pt")
            continue
        pos, opa = result
        elev = elevation(pos)
        data[method] = {
            "opa":   bin_mean(opa, elev),
            "count": bin_count(elev),
        }

    if len(data) < 2:
        return None

    for method in ["v2_spags", "v2_baseline"]:
        if method not in data:
            continue
        opa_arr   = data[method]["opa"]
        norm_arr  = eq_normalize(opa_arr)
        valid     = ~np.isnan(opa_arr)
        ax_opa.plot(LAT_CENTERS[valid], opa_arr[valid],
                    color=COLORS[method], lw=1.5, alpha=0.7, label=LABELS[method])
        ax_norm.plot(LAT_CENTERS[valid], norm_arr[valid],
                     color=COLORS[method], lw=1.5, alpha=0.7)

    ax_opa.set_title(scene_name, fontsize=9)
    ax_norm.set_title(scene_name, fontsize=9)
    return data


def main():
    scenes = list(SCENE_PATHS.keys())
    n = len(scenes)

    fig_opa,  axes_opa  = plt.subplots(2, 4, figsize=(18, 8), sharey=False)
    fig_norm, axes_norm = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
    axes_opa  = axes_opa.flatten()
    axes_norm = axes_norm.flatten()

    # 集約用
    agg_norm_spags    = np.zeros(len(LAT_CENTERS))
    agg_norm_baseline = np.zeros(len(LAT_CENTERS))
    agg_count = np.zeros(len(LAT_CENTERS))

    for idx, scene_name in enumerate(scenes):
        ax_o = axes_opa[idx]
        ax_n = axes_norm[idx]
        result = analyze_single(scene_name, SCENE_PATHS[scene_name], ax_o, ax_n)

        if result is not None:
            for method in ["v2_spags", "v2_baseline"]:
                if method not in result:
                    continue
                norm = eq_normalize(result[method]["opa"])
                valid = ~np.isnan(norm)
                if method == "v2_spags":
                    agg_norm_spags[valid]    += norm[valid]
                else:
                    agg_norm_baseline[valid] += norm[valid]
            agg_count += 1

        for ax in [ax_o, ax_n]:
            ax.axvline(0, color="gray", lw=0.8, ls=":")
            ax.set_xlabel("Latitude [deg]", fontsize=8)
            ax.set_xlim(-90, 90)

    # 凡例（最後の軸に）
    for ax in [axes_opa[-1], axes_norm[-1]]:
        ax.axis("off")
    axes_opa[0].legend(fontsize=8)

    fig_opa.suptitle(
        "Mean Opacity by Latitude\n"
        "Standard L1 vs WS-L1 Training — Training Bias Analysis",
        fontsize=12)
    fig_norm.suptitle(
        "Normalized Mean Opacity (equator=1) by Latitude\n"
        "Standard L1 vs WS-L1 — Training Bias Analysis",
        fontsize=12)

    fig_opa.tight_layout()
    fig_norm.tight_layout()
    fig_opa.savefig(OUTPUT_DIR / "training_bias_opacity.png", dpi=150)
    fig_norm.savefig(OUTPUT_DIR / "training_bias_opacity_norm.png", dpi=150)
    plt.close('all')
    print(f"Saved: {OUTPUT_DIR}/training_bias_opacity.png")
    print(f"Saved: {OUTPUT_DIR}/training_bias_opacity_norm.png")

    # ── 集約プロット ──
    n_safe = agg_count.copy()
    n_safe[n_safe == 0] = 1
    avg_spags    = agg_norm_spags    / n_safe
    avg_baseline = agg_norm_baseline / n_safe

    cos_theory = np.cos(np.radians(LAT_CENTERS)).clip(min=0.01)
    theory_norm = (1 / cos_theory) / (1 / cos_theory[np.abs(LAT_CENTERS) < 15]).mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(LAT_CENTERS, avg_spags,    color=COLORS["v2_spags"],    lw=2.5, label=LABELS["v2_spags"])
    ax.plot(LAT_CENTERS, avg_baseline, color=COLORS["v2_baseline"], lw=2.5, label=LABELS["v2_baseline"])
    ax.plot(LAT_CENTERS, theory_norm,  color="#888888", lw=1.5, ls="--", label="1/cos(θ) theory")
    ax.axhline(1.0, color="#CCCCCC", lw=1, ls=":")
    ax.set_xlabel("Elevation Angle [deg]", fontsize=11)
    ax.set_ylabel("Normalized Mean Opacity (equator=1)", fontsize=10)
    ax.set_title("Aggregated across all scenes\n(Standard L1 inflates polar opacity if bias exists)", fontsize=11)
    ax.legend(fontsize=9)
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlim(-90, 90)
    ax.grid(ls="--", alpha=0.4)

    ax = axes[1]
    diff = avg_spags - avg_baseline
    ax.bar(LAT_CENTERS, diff, width=4.5, color=np.where(diff > 0, "#E05A2B", "#4477AA"), alpha=0.7)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Elevation Angle [deg]", fontsize=11)
    ax.set_ylabel("Δ Normalized Opacity (spags − baseline)", fontsize=10)
    ax.set_title("Opacity difference: Standard L1 − WS-L1\n(positive = Standard L1 has more opacity here)", fontsize=11)
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlim(-90, 90)
    ax.grid(ls="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_bias_aggregate.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR}/training_bias_aggregate.png")

    # ── 数値サマリ ──
    print("\n=== 集約結果 (正規化opacity, 赤道=1.0) ===")
    print(f"{'Lat':>8} | {'Standard L1':>12} | {'WS-L1':>12} | {'Diff':>10}")
    print("-" * 52)
    for i, lat in enumerate(LAT_CENTERS):
        if agg_count[i] == 0:
            continue
        print(f"{lat:>+8.1f}° | {avg_spags[i]:>12.3f} | {avg_baseline[i]:>12.3f} | {diff[i]:>+10.3f}")


if __name__ == "__main__":
    main()
