#!/usr/bin/env python3
"""
analyze_score_bias.py

「opacityスコアは極付近で実際に高く出るか」を検証する。

分析1: baseline最終モデルのopacity × 緯度分布（スコアバイアスの確認）
分析2: opacity pruning vs proposed で緯度ごとの生存率を比較
分析3: 複数シーン集約
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

OUTPUT_DIR = Path("results/score_bias")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAT_BINS = np.linspace(-90, 90, 37)       # 5°刻み
LAT_CENTERS = 0.5 * (LAT_BINS[:-1] + LAT_BINS[1:])

# 解析対象シーン（v2実験、完了済みのみ）
SCENES = {
    "barbershop": {
        "baseline": "output/SPaGS/barbershop_v2_baseline_2026-05-22-12-52-20",
        "opacity":  "output/SPaGS/barbershop_v2_opacity_2026-05-22-17-24-34",
        "proposed": "output/SPaGS/barbershop_v2_proposed_2026-05-22-17-35-08",
    },
    "archiviz-flat": {
        "baseline": "output/SPaGS/archiviz-flat_v2_baseline_2026-05-22-13-46-12",
        "opacity":  "output/SPaGS/archiviz-flat_v2_opacity_2026-05-26-10-42-22",
        "proposed": "output/SPaGS/archiviz-flat_v2_proposed_2026-05-26-10-51-43",
    },
    "bistro_bike": {
        "baseline": "output/SPaGS/bistro_bike_v2_baseline_2026-05-22-14-26-25",
        "opacity":  "output/SPaGS/bistro_bike_v2_opacity_2026-05-22-18-03-06",
        "proposed": "output/SPaGS/bistro_bike_v2_proposed_2026-05-22-18-15-51",
    },
    "bistro_square": {
        "baseline": "output/SPaGS/bistro_square_v2_baseline_2026-05-22-18-41-30",
        "opacity":  "output/SPaGS/bistro_square_v2_opacity_2026-05-22-18-55-10",
        "proposed": "output/SPaGS/bistro_square_v2_proposed_2026-05-22-19-06-54",
    },
    "classroom": {
        "baseline": "output/SPaGS/classroom_v2_baseline_2026-05-22-19-29-46",
        "opacity":  "output/SPaGS/classroom_v2_opacity_2026-05-22-19-40-53",
        "proposed": "output/SPaGS/classroom_v2_proposed_2026-05-22-19-51-37",
    },
    "fisher-hut": {
        "baseline": "output/SPaGS/fisher-hut_v2_baseline_2026-05-22-20-09-24",
        "opacity":  "output/SPaGS/fisher-hut_v2_opacity_2026-05-26-11-09-04",
        "proposed": "output/SPaGS/fisher-hut_v2_proposed_2026-05-26-11-16-22",
    },
    "lone_monk": {
        "baseline": "output/SPaGS/lone_monk_v2_baseline_2026-05-26-11-37-26",
        # v2_opacity / v2_proposed は現在訓練中のためスキップ（final.ptなし時はload_gaussiansがNoneを返す）
    },
}

METHOD_COLORS = {
    "baseline": "#1f77b4",
    "opacity":  "#ff7f0e",
    "proposed": "#2ca02c",
}
METHOD_LABELS = {
    "baseline": "No pruning (baseline)",
    "opacity":  "Opacity pruning",
    "proposed": "Proposed (region-normalized)",
}


# ──────────────────────────────────────────────
# ユーティリティ
# ──────────────────────────────────────────────

def load_gaussians(run_dir: str) -> dict | None:
    """チェックポイントからGaussianパラメータを読み込む。"""
    pt_path = Path(run_dir) / "checkpoints" / "final.pt"
    if not pt_path.exists():
        print(f"  [SKIP] {pt_path} not found")
        return None
    pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = pt["model_state_dict"]
    return {
        "positions": sd["gaussians._positions"].numpy(),   # [N, 3]
        "opacities": sd["gaussians._opacities"].numpy(),   # [N, 1]  baked=activated
        "scales":    sd["gaussians._scales"].numpy(),      # [N, 3]  baked=activated
    }


def compute_elevation(positions: np.ndarray) -> np.ndarray:
    """Y上向き座標系での仰角[度] θ = atan2(y, sqrt(x²+z²))"""
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    dist_xz = np.sqrt(x**2 + z**2).clip(min=1e-6)
    return np.degrees(np.arctan2(y, dist_xz))   # [-90, 90]


def bin_by_latitude(values: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
    """緯度ビンごとのmean。値がないビンはnan。"""
    result = np.full(len(LAT_CENTERS), np.nan)
    for i, (lo, hi) in enumerate(zip(LAT_BINS[:-1], LAT_BINS[1:])):
        mask = (elevation_deg >= lo) & (elevation_deg < hi)
        if mask.sum() > 0:
            result[i] = values[mask].mean()
    return result


def count_by_latitude(elevation_deg: np.ndarray) -> np.ndarray:
    """緯度ビンごとのGaussian数。"""
    counts = np.zeros(len(LAT_CENTERS))
    for i, (lo, hi) in enumerate(zip(LAT_BINS[:-1], LAT_BINS[1:])):
        counts[i] = ((elevation_deg >= lo) & (elevation_deg < hi)).sum()
    return counts


def ideal_uniform_sphere(lat_centers_deg: np.ndarray) -> np.ndarray:
    """一様球面上の緯度分布（cos(θ)に比例）。正規化して返す。"""
    vals = np.cos(np.radians(lat_centers_deg)).clip(min=0)
    return vals / vals.sum()


# ──────────────────────────────────────────────
# 分析1: scoreの緯度バイアス（1シーン）
# ──────────────────────────────────────────────

def analyze_score_bias_single(scene_name: str, paths: dict) -> None:
    print(f"\n=== 分析1: score bias [{scene_name}] ===")

    baseline = load_gaussians(paths["baseline"])
    if baseline is None:
        return

    elev = compute_elevation(baseline["positions"])
    opacity = baseline["opacities"].squeeze()
    vol = np.prod(baseline["scales"], axis=1)   # s0*s1*s2（体積に比例）

    # 体積正規化スコア（単純な contribution score の近似）
    v90 = np.quantile(vol, 0.90).clip(min=1e-9)
    v_norm = (vol / v90).clip(max=1.0)
    gamma = v_norm ** 0.5
    naive_score = opacity * gamma                        # 仰角補正なし
    corrected_score = opacity * gamma * np.cos(np.radians(elev)).clip(min=0.1)  # cos補正あり

    opacity_by_lat    = bin_by_latitude(opacity, elev)
    naive_by_lat      = bin_by_latitude(naive_score, elev)
    corrected_by_lat  = bin_by_latitude(corrected_score, elev)
    count_by_lat      = count_by_latitude(elev)

    # ── プロット ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Score Bias by Latitude — {scene_name} (baseline final model)", fontsize=12)

    # (a) Gaussian数の緯度分布
    ax = axes[0]
    ax.bar(LAT_CENTERS, count_by_lat, width=5, color="steelblue", alpha=0.7, label="Actual")
    ideal = ideal_uniform_sphere(LAT_CENTERS) * count_by_lat.sum()
    ax.plot(LAT_CENTERS, ideal, "r--", lw=2, label="Uniform sphere (cos θ)")
    ax.set_xlabel("Latitude [deg]")
    ax.set_ylabel("Gaussian count")
    ax.set_title("(a) Gaussian distribution by latitude")
    ax.legend(fontsize=8)
    ax.axvline(0, color="gray", lw=0.8, ls=":")

    # (b) opacity の緯度分布
    ax = axes[1]
    ax.plot(LAT_CENTERS, opacity_by_lat, "o-", color="tab:orange", lw=2, ms=4)
    ax.set_xlabel("Latitude [deg]")
    ax.set_ylabel("Mean opacity")
    ax.set_title("(b) Mean opacity by latitude\n(high at poles = biased toward poles)")
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.axhline(np.nanmean(opacity_by_lat), color="gray", lw=1, ls="--", label="Overall mean")
    ax.legend(fontsize=8)

    # (c) naiveスコア vs cos補正スコアの比較
    ax = axes[2]
    # nanを除いて正規化（比較しやすくする）
    valid = ~np.isnan(naive_by_lat) & ~np.isnan(corrected_by_lat)
    n_mean = np.nanmean(naive_by_lat)
    c_mean = np.nanmean(corrected_by_lat)
    ax.plot(LAT_CENTERS[valid], naive_by_lat[valid] / n_mean,
            "o-", color="tab:red", lw=2, ms=4, label="w/o cos correction (naive)")
    ax.plot(LAT_CENTERS[valid], corrected_by_lat[valid] / c_mean,
            "s-", color="tab:green", lw=2, ms=4, label="w/ cos(θ) correction")
    ax.axhline(1.0, color="gray", lw=1, ls="--", label="Mean=1 baseline")
    ax.set_xlabel("Latitude [deg]")
    ax.set_ylabel("Normalized score (mean=1)")
    ax.set_title("(c) Score w/ vs w/o elevation correction\n(naive score inflated at poles)")
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = OUTPUT_DIR / f"score_bias_{scene_name}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → saved: {out}")


# ──────────────────────────────────────────────
# 分析2: 緯度ごとの生存率（opacity vs proposed）
# ──────────────────────────────────────────────

def analyze_survival_rate(scene_name: str, paths: dict) -> None:
    print(f"\n=== 分析2: 生存率 [{scene_name}] ===")

    data = {}
    for method in ["baseline", "opacity", "proposed"]:
        if method not in paths:
            continue
        g = load_gaussians(paths[method])
        if g is None:
            continue
        elev = compute_elevation(g["positions"])
        data[method] = count_by_latitude(elev)

    if "baseline" not in data:
        print("  baseline がないのでスキップ")
        return

    base_count = data["baseline"]
    # ゼロ除算を避ける
    base_safe = base_count.copy()
    base_safe[base_safe == 0] = 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Gaussian survival by latitude — {scene_name}", fontsize=12)

    # (a) 絶対Gaussian数の比較
    ax = axes[0]
    for method, cnt in data.items():
        ax.plot(LAT_CENTERS, cnt, "o-", color=METHOD_COLORS[method],
                lw=2, ms=3, label=METHOD_LABELS[method])
    ax.set_xlabel("Latitude [deg]")
    ax.set_ylabel("Gaussian count")
    ax.set_title("(a) Gaussian count by latitude\nafter training / pruning")
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.legend(fontsize=8)

    # (b) 各手法のbaseline比 survival rate
    ax = axes[1]
    for method, cnt in data.items():
        if method == "baseline":
            continue
        survival = cnt / base_safe
        ax.plot(LAT_CENTERS, survival, "o-", color=METHOD_COLORS[method],
                lw=2, ms=3, label=METHOD_LABELS[method])
    # 理想（一様生存）
    ax.axhline(1.0, color="gray", lw=1, ls="--", label="No pruning (baseline)")
    ax.set_xlabel("Latitude [deg]")
    ax.set_ylabel("Survival ratio vs baseline")
    ax.set_title("(b) Survival ratio vs baseline\n"
                 "opacity pruning over-keeps poles → proposed fixes this")
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = OUTPUT_DIR / f"survival_rate_{scene_name}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → saved: {out}")

    # 数値サマリ
    print(f"\n  {'Latitude':>10} | {'Baseline':>10} | {'Opacity':>10} | {'Proposed':>10} | Opacity/Prop")
    print("  " + "-" * 65)
    for i, lat in enumerate(LAT_CENTERS):
        base = int(base_count[i])
        op   = int(data.get("opacity",  np.zeros_like(base_count))[i])
        prop = int(data.get("proposed", np.zeros_like(base_count))[i])
        ratio = (op / prop) if prop > 0 else float("nan")
        print(f"  {lat:>+10.1f}° | {base:>10,} | {op:>10,} | {prop:>10,} | {ratio:>10.2f}")


# ──────────────────────────────────────────────
# 分析3: 複数シーン集約
# ──────────────────────────────────────────────

def analyze_aggregate(scenes: dict) -> None:
    print("\n=== 分析3: 複数シーン集約 ===")

    agg_opacity_bias  = np.zeros(len(LAT_CENTERS))
    agg_survival_op   = np.zeros(len(LAT_CENTERS))
    agg_survival_prop = np.zeros(len(LAT_CENTERS))
    n_valid = np.zeros(len(LAT_CENTERS))

    for scene_name, paths in scenes.items():
        base_g = load_gaussians(paths["baseline"])
        op_g   = load_gaussians(paths.get("opacity", ""))
        prop_g = load_gaussians(paths.get("proposed", ""))
        if base_g is None:
            continue

        base_elev  = compute_elevation(base_g["positions"])
        base_cnt   = count_by_latitude(base_elev)
        base_safe  = base_cnt.copy(); base_safe[base_safe == 0] = 1

        # opacity bias
        opacity = base_g["opacities"].squeeze()
        op_lat = bin_by_latitude(opacity, base_elev)
        mean_op = np.nanmean(op_lat)
        if mean_op > 0:
            agg_opacity_bias += np.nan_to_num(op_lat / mean_op)

        # survival
        if op_g is not None:
            op_cnt  = count_by_latitude(compute_elevation(op_g["positions"]))
            agg_survival_op += op_cnt / base_safe
        if prop_g is not None:
            prop_cnt = count_by_latitude(compute_elevation(prop_g["positions"]))
            agg_survival_prop += prop_cnt / base_safe

        n_valid += 1

    n_valid_safe = n_valid.copy(); n_valid_safe[n_valid_safe == 0] = 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Aggregated bias analysis (all scenes)", fontsize=12)

    ax = axes[0]
    ax.plot(LAT_CENTERS, agg_opacity_bias / n_valid_safe, "o-", color="tab:orange", lw=2, ms=4)
    ax.axhline(1.0, color="gray", lw=1, ls="--", label="Unbiased (mean=1)")
    ax.set_xlabel("Latitude [deg]")
    ax.set_ylabel("Normalized opacity (mean=1)")
    ax.set_title("(a) Opacity score bias by latitude\n(>1 at poles → biased)")
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(LAT_CENTERS, agg_survival_op / n_valid_safe, "o-",
            color=METHOD_COLORS["opacity"], lw=2, ms=4, label=METHOD_LABELS["opacity"])
    ax.plot(LAT_CENTERS, agg_survival_prop / n_valid_safe, "s-",
            color=METHOD_COLORS["proposed"], lw=2, ms=4, label=METHOD_LABELS["proposed"])
    ax.axhline(1.0, color="gray", lw=1, ls="--", label="Baseline survival")
    ax.set_xlabel("Latitude [deg]")
    ax.set_ylabel("Survival ratio vs baseline")
    ax.set_title("(b) Survival rate by latitude\n(opacity pruning over-keeps poles)")
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = OUTPUT_DIR / "aggregate_bias.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → saved: {out}")


# ──────────────────────────────────────────────
# エントリポイント
# ──────────────────────────────────────────────

if __name__ == "__main__":
    for scene_name, paths in SCENES.items():
        analyze_score_bias_single(scene_name, paths)
        analyze_survival_rate(scene_name, paths)
    analyze_aggregate(SCENES)
    print(f"\n全完了。結果: {OUTPUT_DIR}/")
