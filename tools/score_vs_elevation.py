#!/usr/bin/env python3
"""ラスタライザの実スコア（max α·T の平均）を仰角別に集計して理論値と比較する。"""
import os
os.environ["WANDB_MODE"] = "disabled"
import sys
import math
import random
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


def run_scene(scene_name, exp_dir):
    import shutil
    import utils
    with utils.discoverSourcePath():
        import Framework
        from Implementations import Methods as MI, Datasets as DI

    Framework.setup(config_path=f"{exp_dir}/training_config.yaml", require_custom_config=True)
    # Delete the spurious output directory Framework creates for analysis runs
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

    dataset = DI.getDataset(dataset_type=Framework.config.GLOBAL.DATASET_TYPE, path=Framework.config.DATASET.PATH)
    dataset.setMode("train")
    renderer = model_instance.renderer

    positions = g.get_positions.detach()
    scales    = g.get_scales.detach()
    rotations = g.get_rotations.detach()
    opacities = g.get_opacities.detach()
    n = positions.shape[0]

    dist_xz  = torch.norm(positions[:, [0, 2]], dim=1).clamp(1e-6)
    theta     = torch.atan2(positions[:, 1], dist_xz)
    elev_deg  = (theta * 180 / math.pi).cpu().numpy()
    opa_cpu   = opacities.squeeze().cpu().numpy()

    random.seed(42)
    all_props = list(dataset.data[dataset.mode])
    sampled   = random.sample(all_props, min(N_FRAMES, len(all_props)))

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

    avg_w = (scores_sum / len(sampled)).cpu().numpy()
    Framework.teardown()

    bins    = np.linspace(-90, 90, 19)
    centers = 0.5 * (bins[:-1] + bins[1:])
    cos_th  = np.cos(np.radians(centers)).clip(0.05)

    eq_w, eq_o = [], []
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        m = (elev_deg >= lo) & (elev_deg < hi) & (avg_w > 0)
        if m.sum() >= 3 and abs(centers[i]) < 15:
            eq_w.append(avg_w[m].mean())
            eq_o.append(opa_cpu[m].mean())
    eq_w_mean = np.mean(eq_w) if eq_w else 1.0
    eq_o_mean = np.mean(eq_o) if eq_o else 1.0
    theory_eq = np.mean(1 / cos_th[np.abs(centers) < 15])

    print(f"\n=== {scene_name} (N={n:,}) ===")
    print(f"{'Lat':>8} | {'N':>6} | {'AvgMaxW':>9} | {'NormW':>8} | {'NormOpa':>8} | {'Theory':>8}")
    print("-" * 62)
    norm_w_list = []
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        m = (elev_deg >= lo) & (elev_deg < hi) & (avg_w > 0)
        cnt = m.sum()
        if cnt < 3:
            norm_w_list.append(None)
            continue
        aw   = avg_w[m].mean()
        oa   = opa_cpu[m].mean()
        nw   = aw / eq_w_mean
        no   = oa / eq_o_mean
        th   = (1 / cos_th[i]) / theory_eq
        print(f"{centers[i]:>+8.1f}° | {cnt:>6,} | {aw:>9.5f} | {nw:>8.3f} | {no:>8.3f} | {th:>8.3f}")
        norm_w_list.append(nw)

    print(f"  赤道基準 mean max_weight = {eq_w_mean:.5f}")
    return centers, np.array([x if x is not None else np.nan for x in norm_w_list])


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results = {}
    for scene_name, exp_dir in SCENES.items():
        if not Path(f"{exp_dir}/checkpoints/final.pt").exists():
            print(f"[SKIP] {scene_name}: no final.pt")
            continue
        print(f"\nProcessing {scene_name}...")
        centers, norm_w = run_scene(scene_name, exp_dir)
        results[scene_name] = (centers, norm_w)

    if not results:
        print("No results.")
        return

    # aggregate
    all_norm = np.stack([v for _, v in results.values()], axis=0)
    agg = np.nanmean(all_norm, axis=0)
    first_centers = list(results.values())[0][0]

    cos_th  = np.cos(np.radians(first_centers)).clip(0.05)
    theory  = (1 / cos_th) / np.mean(1 / cos_th[np.abs(first_centers) < 15])

    print(f"\n=== 全シーン平均 ===")
    print(f"{'Lat':>8} | {'NormW(avg)':>10} | {'Theory':>8}")
    print("-" * 35)
    for i, lat in enumerate(first_centers):
        if not np.isnan(agg[i]):
            print(f"{lat:>+8.1f}° | {agg[i]:>10.3f} | {theory[i]:>8.3f}")

    # plot
    out = Path("results/figures/score_vs_elevation_agg.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for scene_name, (centers, norm_w) in results.items():
        ax.plot(centers, norm_w, lw=1.2, alpha=0.6, label=scene_name)
    ax.plot(first_centers, agg,    color='black', lw=2.5, label='Mean (all scenes)')
    ax.plot(first_centers, theory, color='blue',  lw=2,   linestyle='--', label='1/cos(θ) theory')
    ax.axhline(1.0, color='gray', lw=0.8, linestyle=':')
    ax.axvspan(-90, -60, alpha=0.06, color='blue')
    ax.axvspan(60,   90, alpha=0.06, color='blue')
    ax.set_xlabel('Elevation [deg]')
    ax.set_ylabel('Normalized max α·T (equator=1)')
    ax.set_title('Score Bias: avg max α·T per Gaussian vs Elevation\n(via rasterizer, v2_spags checkpoints)')
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(-90, 90)
    ax.grid(linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved → {out}")


if __name__ == '__main__':
    main()
