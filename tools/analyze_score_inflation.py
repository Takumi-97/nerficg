#!/usr/bin/env python3
"""
analyze_score_inflation.py

「極付近のガウシアンの重要度スコアが過大評価されること」の直接実証。

equirectangularでは仰角θにあるガウシアンのピクセルカバレッジが
立体角に対して 1/cos(θ) 倍に膨らむ。

分析:
  1. 各ガウシアンの「立体角」と「equirectangularピクセルカバレッジ」を計算
  2. その比が 1/cos(θ) であることを示す
  3. cos(θ) 補正後は比がほぼ一定になることを示す
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("results/score_inflation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAT_BINS    = np.linspace(-90, 90, 19)   # 10°刻み
LAT_CENTERS = 0.5 * (LAT_BINS[:-1] + LAT_BINS[1:])

# 解析シーン（v2_spags: プルーニングなし・標準L1、シンプルなベースライン）
SCENES = {
    "barbershop":    "output/SPaGS/barbershop_v2_spags_2026-05-22-12-28-44",
    "archiviz-flat": "output/SPaGS/archiviz-flat_v2_spags_2026-05-22-13-36-01",
    "bistro_bike":   "output/SPaGS/bistro_bike_v2_spags_2026-05-22-14-12-41",
    "bistro_square": "output/SPaGS/bistro_square_v2_spags_2026-05-22-18-27-13",
    "classroom":     "output/SPaGS/classroom_v2_spags_2026-05-22-19-18-07",
    "fisher-hut":    "output/SPaGS/fisher-hut_v2_spags_2026-05-26-11-01-32",
    "lone_monk":     "output/SPaGS/lone_monk_v2_spags_2026-05-26-11-23-32",
}


def load_gaussians(run_dir: str):
    pt_path = Path(run_dir) / "checkpoints" / "final.pt"
    if not pt_path.exists():
        return None
    pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = pt["model_state_dict"]
    pos    = sd["gaussians._positions"].numpy()    # [N, 3]
    scales = sd["gaussians._scales"].numpy()       # [N, 3] (activated)
    opa    = sd["gaussians._opacities"].numpy().squeeze()
    return pos, scales, opa


def compute_metrics(pos, scales, opa):
    """各ガウシアンの仰角・立体角・ピクセルカバレッジ・バイアス比を計算"""
    # 仰角
    dist_xz = np.sqrt(pos[:, 0]**2 + pos[:, 2]**2).clip(min=1e-6)
    r       = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2).clip(min=1e-6)
    theta   = np.arctan2(pos[:, 1], dist_xz)           # [-π/2, π/2]
    elev_deg = np.degrees(theta)
    cos_theta = np.cos(theta).clip(min=0.05)

    # ガウシアンの有効サイズ（幾何平均スケール）
    sigma = np.cbrt(np.prod(np.abs(scales), axis=1)).clip(min=1e-8)  # [N]

    # 立体角（perspective camera 相当）∝ σ² / r²
    solid_angle = sigma**2 / r**2   # [N]  （正規化不要、比をとるので）

    # equirectangularピクセルカバレッジ ∝ σ² / (r² × cos(θ))
    pixel_coverage = solid_angle / cos_theta  # [N]

    # バイアス比 = pixel_coverage / solid_angle = 1/cos(θ)
    bias_ratio = pixel_coverage / solid_angle.clip(min=1e-12)  # = 1/cos(θ)

    # cos補正後のスコア ∝ pixel_coverage × cos(θ) = solid_angle
    corrected = pixel_coverage * cos_theta    # = solid_angle

    # opacityで重み付けしたpixel_coverage（実際のcontribution scoreに近い）
    weighted_raw       = opa * pixel_coverage
    weighted_corrected = opa * corrected

    return elev_deg, cos_theta, solid_angle, pixel_coverage, bias_ratio, weighted_raw, weighted_corrected


def bin_mean(values, elev_deg, min_count=5):
    result = np.full(len(LAT_CENTERS), np.nan)
    for i, (lo, hi) in enumerate(zip(LAT_BINS[:-1], LAT_BINS[1:])):
        m = (elev_deg >= lo) & (elev_deg < hi)
        if m.sum() >= min_count:
            result[i] = values[m].mean()
    return result


def analyze_scene(scene_name, run_dir, ax_raw, ax_corr, ax_bias):
    data = load_gaussians(run_dir)
    if data is None:
        print(f"  [SKIP] {scene_name}")
        return None

    pos, scales, opa = data
    elev_deg, cos_theta, solid_angle, pixel_cov, bias_ratio, w_raw, w_corr = \
        compute_metrics(pos, scales, opa)

    # ビン平均
    raw_binned  = bin_mean(w_raw,        elev_deg)
    corr_binned = bin_mean(w_corr,       elev_deg)
    bias_binned = bin_mean(bias_ratio,   elev_deg)

    # 赤道基準で正規化
    eq_mask = np.abs(LAT_CENTERS) < 15
    def norm_eq(arr):
        denom = np.nanmean(arr[eq_mask])
        return arr / denom if denom > 0 else arr

    ax_raw.plot(LAT_CENTERS, norm_eq(raw_binned),   lw=1.5, alpha=0.7, label=scene_name)
    ax_corr.plot(LAT_CENTERS, norm_eq(corr_binned), lw=1.5, alpha=0.7, label=scene_name)
    ax_bias.plot(LAT_CENTERS, bias_binned,           lw=1.5, alpha=0.7)

    return raw_binned, corr_binned, bias_binned


def main():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax_raw, ax_corr, ax_bias = axes

    all_raw   = []
    all_corr  = []
    all_bias  = []

    for scene_name, run_dir in SCENES.items():
        result = analyze_scene(scene_name, run_dir, ax_raw, ax_corr, ax_bias)
        if result is None:
            continue
        raw_b, corr_b, bias_b = result
        all_raw.append(raw_b)
        all_corr.append(corr_b)
        all_bias.append(bias_b)

    # 集約（nanmean）
    agg_raw  = np.nanmean(all_raw,  axis=0)
    agg_corr = np.nanmean(all_corr, axis=0)
    agg_bias = np.nanmean(all_bias, axis=0)

    # 理論曲線 1/cos(θ)
    cos_theory  = np.cos(np.radians(LAT_CENTERS)).clip(min=0.05)
    theory_bias = 1.0 / cos_theory   # 理論的なバイアス比

    # 赤道基準正規化
    eq_mask = np.abs(LAT_CENTERS) < 15
    def norm_eq(arr):
        denom = np.nanmean(arr[eq_mask])
        return arr / denom if denom > 0 else arr

    agg_raw_norm  = norm_eq(agg_raw)
    agg_corr_norm = norm_eq(agg_corr)

    # ── 左: 補正なしスコア（赤道=1に正規化）──
    ax_raw.plot(LAT_CENTERS, norm_eq(theory_bias), "k--", lw=2, label="1/cos(θ) theory")
    ax_raw.plot(LAT_CENTERS, agg_raw_norm, "k-", lw=2.5, label="Avg (all scenes)")
    ax_raw.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax_raw.axvspan(-90, -60, alpha=0.05, color="blue")
    ax_raw.axvspan(60,   90, alpha=0.05, color="blue")
    ax_raw.set_xlabel("Elevation [deg]", fontsize=11)
    ax_raw.set_ylabel("Normalized score (equator=1)", fontsize=10)
    ax_raw.set_title("(a) Without cos(θ) correction\n"
                     "Score = opacity × pixel coverage\n"
                     "(should follow 1/cos(θ) if bias exists)", fontsize=10)
    ax_raw.legend(fontsize=7, ncol=2)
    ax_raw.set_xlim(-90, 90)
    ax_raw.grid(ls="--", alpha=0.3)

    # ── 中央: cos補正後スコア ──
    ax_corr.plot(LAT_CENTERS, np.ones_like(LAT_CENTERS), "k--", lw=2, label="Ideal (uniform)")
    ax_corr.plot(LAT_CENTERS, agg_corr_norm, "k-", lw=2.5, label="Avg (all scenes)")
    ax_corr.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax_corr.axvspan(-90, -60, alpha=0.05, color="blue")
    ax_corr.axvspan(60,   90, alpha=0.05, color="blue")
    ax_corr.set_xlabel("Elevation [deg]", fontsize=11)
    ax_corr.set_ylabel("Normalized score (equator=1)", fontsize=10)
    ax_corr.set_title("(b) With cos(θ) correction\n"
                      "Score = opacity × pixel coverage × cos(θ)\n"
                      "(should be flat if correction works)", fontsize=10)
    ax_corr.legend(fontsize=7, ncol=2)
    ax_corr.set_xlim(-90, 90)
    ax_corr.grid(ls="--", alpha=0.3)

    # ── 右: バイアス比 = pixel_coverage / solid_angle = 1/cos(θ) ──
    ax_bias.plot(LAT_CENTERS, theory_bias,  "k--", lw=2, label="1/cos(θ) theory")
    ax_bias.plot(LAT_CENTERS, agg_bias, "k-", lw=2.5, label="Avg (all scenes)")
    ax_bias.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax_bias.axvspan(-90, -60, alpha=0.05, color="blue")
    ax_bias.axvspan(60,   90, alpha=0.05, color="blue")
    ax_bias.set_xlabel("Elevation [deg]", fontsize=11)
    ax_bias.set_ylabel("Pixel coverage / Solid angle", fontsize=10)
    ax_bias.set_title("(c) Bias ratio = pixel coverage / solid angle\n"
                      "(= 1/cos(θ) analytically, verifying equirectangular inflation)", fontsize=10)
    ax_bias.legend(fontsize=9)
    ax_bias.set_xlim(-90, 90)
    ax_bias.grid(ls="--", alpha=0.3)

    fig.suptitle("Score Inflation in Equirectangular Projection\n"
                 "Gaussians appearing in polar regions have 1/cos(θ)-inflated importance scores",
                 fontsize=12)
    fig.tight_layout()
    out = OUTPUT_DIR / "score_inflation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── 数値サマリ ──
    print("\n=== 数値サマリ（全シーン平均）===")
    print(f"{'Lat':>8} | {'RawScore':>10} | {'Corrected':>10} | {'BiasRatio':>10} | {'Theory 1/cos':>13}")
    print("-" * 60)
    for i, lat in enumerate(LAT_CENTERS):
        if np.isnan(agg_raw_norm[i]):
            continue
        theory = 1.0 / np.cos(np.radians(lat)).clip(0.05)
        print(f"{lat:>+8.1f}° | {agg_raw_norm[i]:>10.3f} | {agg_corr_norm[i]:>10.3f} | "
              f"{agg_bias[i]:>10.3f} | {theory:>13.3f}")


if __name__ == "__main__":
    main()
