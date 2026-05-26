#!/usr/bin/env python3
"""
補正あり（camcos）vs 補正なし（raw）でGaussianの重要度スコアを仰角別に比較する。

spags（補正なし訓練）とproposed_camcos（補正あり訓練）の両チェックポイントに対し、
それぞれ update_max_weights を呼んでスコア分布を計算する。
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
    "barbershop":    ("output/SPaGS/barbershop_v2_spags_2026-05-22-12-28-44",
                      "output/SPaGS/barbershop_proposed_camcos_2026-05-22-14-43-48"),
    "archiviz-flat": ("output/SPaGS/archiviz-flat_v2_spags_2026-05-22-13-36-01",
                      "output/SPaGS/archiviz-flat_proposed_camcos_2026-05-22-14-55-55"),
    "bistro_bike":   ("output/SPaGS/bistro_bike_v2_spags_2026-05-22-14-12-41",
                      "output/SPaGS/bistro_bike_proposed_camcos_2026-05-22-15-09-14"),
    "classroom":     ("output/SPaGS/classroom_v2_spags_2026-05-22-19-18-07",
                      "output/SPaGS/classroom_proposed_camcos_2026-05-22-15-41-02"),
    "fisher-hut":    ("output/SPaGS/fisher-hut_v2_spags_2026-05-26-11-01-32",
                      "output/SPaGS/fisher-hut_proposed_camcos_2026-05-22-15-59-34"),
    "lone_monk":     ("output/SPaGS/lone_monk_v2_spags_2026-05-26-11-23-32",
                      "output/SPaGS/lone_monk_proposed_camcos_2026-05-22-16-10-29"),
}

N_FRAMES = 50
BINS = np.linspace(-90, 90, 19)
CENTERS = 0.5 * (BINS[:-1] + BINS[1:])


def load_and_score(exp_dir, n_frames, use_cam_spherical):
    """チェックポイントをロードして仰角ごとの平均スコアを返す。"""
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

    positions  = g.get_positions.detach()
    scales     = g.get_scales.detach()
    rotations  = g.get_rotations.detach()
    opacities  = g.get_opacities.detach()
    n = positions.shape[0]

    # ワールド座標の仰角
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
        if use_cam_spherical:
            T_cam = props.T
            cam_pos = (T_cam.float() if isinstance(T_cam, torch.Tensor)
                       else torch.tensor(T_cam, dtype=torch.float32)).to("cuda")
            dirs     = positions - cam_pos.unsqueeze(0)
            dist_xz2 = torch.norm(dirs[:, [0, 2]], dim=1).clamp(1e-6)
            theta_c  = torch.atan2(dirs[:, 1], dist_xz2)
            cos_cam  = torch.cos(theta_c).clamp(min=0.1)
            fw = fw * cos_cam
        scores_sum += fw

    Framework.teardown()

    avg_score = (scores_sum / len(sampled)).cpu().numpy()
    return elev_deg, avg_score


def bin_mean_norm(elev_deg, scores):
    """仰角ビン平均 → 赤道基準正規化"""
    binned = np.full(len(CENTERS), np.nan)
    for i, (lo, hi) in enumerate(zip(BINS[:-1], BINS[1:])):
        m = (elev_deg >= lo) & (elev_deg < hi) & (scores > 0)
        if m.sum() >= 3:
            binned[i] = scores[m].mean()
    eq = np.nanmean(binned[np.abs(CENTERS) < 15])
    return binned / eq if eq > 0 else binned


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    raw_all, cor_all = [], []

    for scene_name, (spags_dir, camcos_dir) in SCENES.items():
        for d in [spags_dir, camcos_dir]:
            if not Path(f"{d}/checkpoints/final.pt").exists():
                print(f"[SKIP] {scene_name}: no final.pt in {d}")
                continue

        print(f"\n--- {scene_name}: computing raw scores (spags model, no correction) ---")
        elev, raw_scores = load_and_score(spags_dir, N_FRAMES, use_cam_spherical=False)
        raw_norm = bin_mean_norm(elev, raw_scores)

        print(f"--- {scene_name}: computing corrected scores (spags model + cam correction) ---")
        _, cor_scores = load_and_score(spags_dir, N_FRAMES, use_cam_spherical=True)
        cor_norm = bin_mean_norm(elev, cor_scores)

        raw_all.append(raw_norm)
        cor_all.append(cor_norm)

        print(f"\n{scene_name}:")
        print(f"  {'Lat':>7} | {'Raw(no corr)':>12} | {'Corrected':>10}")
        for i, lat in enumerate(CENTERS):
            if not np.isnan(raw_norm[i]):
                print(f"  {lat:>+7.1f}° | {raw_norm[i]:>12.3f} | {cor_norm[i]:>10.3f}")

    # aggregate
    agg_raw = np.nanmean(raw_all, axis=0)
    agg_cor = np.nanmean(cor_all, axis=0)
    cos_th  = np.cos(np.radians(CENTERS)).clip(0.05)
    theory  = (1 / cos_th) / np.mean(1 / cos_th[np.abs(CENTERS) < 15])

    print(f"\n=== 全シーン平均 ===")
    print(f"{'Lat':>8} | {'Raw':>8} | {'Corrected':>10} | {'Theory 1/cos':>13}")
    print("-" * 48)
    for i, lat in enumerate(CENTERS):
        if not np.isnan(agg_raw[i]):
            print(f"{lat:>+8.1f}° | {agg_raw[i]:>8.3f} | {agg_cor[i]:>10.3f} | {theory[i]:>13.3f}")

    # plot
    out = Path("results/figures/score_correction_effect.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左：全シーンの生スコア vs 補正後スコア（aggregate）
    ax = axes[0]
    ax.plot(CENTERS, agg_raw, color='#E05A2B', lw=2.5, label='Without correction (raw)')
    ax.plot(CENTERS, agg_cor, color='#2CA02C', lw=2.5, label='With cam cos(θ) correction')
    ax.plot(CENTERS, theory,  color='#4477AA', lw=1.5, ls='--', label='1/cos(θ) theory')
    ax.axhline(1.0, color='gray', lw=0.8, ls=':')
    ax.axvspan(-90, -60, alpha=0.06, color='blue')
    ax.axvspan(60,   90, alpha=0.06, color='blue')
    ax.set_xlabel('Elevation [deg]')
    ax.set_ylabel('Normalized score (equator=1)')
    ax.set_title('Aggregate (all scenes)\nRaw vs Corrected score by elevation')
    ax.legend(fontsize=9)
    ax.set_xlim(-90, 90)
    ax.grid(ls='--', alpha=0.4)

    # 右：補正によるスコア変化量（correction ratio = corrected/raw）
    ax = axes[1]
    for raw_n, cor_n in zip(raw_all, cor_all):
        ratio = cor_n / np.where(raw_n > 0, raw_n, np.nan)
        ax.plot(CENTERS, ratio, lw=1.2, alpha=0.5)
    agg_ratio = np.nanmean([c / np.where(r > 0, r, np.nan)
                             for r, c in zip(raw_all, cor_all)], axis=0)
    ax.plot(CENTERS, agg_ratio, color='black', lw=2.5, label='Mean ratio')
    ax.axhline(1.0, color='gray', lw=1, ls='--', label='No change')
    ax.set_xlabel('Elevation [deg]')
    ax.set_ylabel('Corrected / Raw score')
    ax.set_title('Effect of cos(θ) correction\n(values < 1 → correction reduced score)')
    ax.legend(fontsize=9)
    ax.set_xlim(-90, 90)
    ax.grid(ls='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved → {out}")


if __name__ == '__main__':
    main()
