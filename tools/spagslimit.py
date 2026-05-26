import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os


# =========================
# 解析本体
# =========================
def spags_diagnosis(xyz, output_dir="results/spags_diagnosis"):

    os.makedirs(output_dir, exist_ok=True)

    center = np.mean(xyz, axis=0)
    xyz_shift = xyz - center

    x, y, z = xyz_shift[:,0], xyz_shift[:,1], xyz_shift[:,2]

    r = np.linalg.norm(xyz_shift, axis=1)
    theta = np.arctan2(y, x)
    phi = np.arcsin(z / (r + 1e-8))

    # =========================================================
    # ① 距離分布チェック（あなたが見つけたやつ）
    # =========================================================
    plt.figure()
    plt.hist(r, bins=200)
    plt.title("Radial Distribution (r)")
    plt.xlabel("r")
    plt.ylabel("count")
    plt.savefig(f"{output_dir}/radial_hist.png")
    plt.close()

    r_peak = np.mean(r)
    print(f"📌 Mean radius: {r_peak:.4f}")

    # =========================================================
    # ② 球面密度（極問題）
    # =========================================================
    phi_hist, phi_bins = np.histogram(phi, bins=180, density=True)

    plt.figure()
    plt.plot(phi_bins[:-1], phi_hist)
    plt.title("Elevation Distribution")
    plt.xlabel("phi [rad]")
    plt.ylabel("density")
    plt.savefig(f"{output_dir}/phi_dist.png")
    plt.close()

    # entropy（偏りの定量化）
    eps = 1e-8
    entropy = -np.sum(phi_hist * np.log(phi_hist + eps))
    print(f"📌 Elevation entropy: {entropy:.4f}")

    # =========================================================
    # ③ 球面方向のカバレッジ
    # =========================================================
    theta_bins = np.linspace(-np.pi, np.pi, 90)
    phi_bins = np.linspace(-np.pi/2, np.pi/2, 45)

    H, _, _ = np.histogram2d(theta, phi, bins=[theta_bins, phi_bins])

    coverage = np.count_nonzero(H) / H.size
    print(f"📌 Spherical coverage ratio: {coverage:.4f}")

    plt.figure()
    plt.imshow(H.T, origin="lower", aspect="auto")
    plt.title("Spherical Occupancy")
    plt.savefig(f"{output_dir}/spherical_occupancy.png")
    plt.close()

    # =========================================================
    # ④ 高周波性（局所変化）
    # =========================================================
    tree = cKDTree(xyz_shift)

    k = 8
    dists, _ = tree.query(xyz_shift, k=k)

    local_variation = np.std(dists[:,1:], axis=1)

    plt.figure()
    plt.hist(local_variation, bins=200)
    plt.title("Local Frequency / Detail Measure")
    plt.xlabel("variation")
    plt.savefig(f"{output_dir}/high_frequency.png")
    plt.close()

    hf_mean = np.mean(local_variation)
    print(f"📌 High-frequency proxy (std NN dist): {hf_mean:.6f}")

    # =========================================================
    # ⑤ 極領域の密度比較
    # =========================================================
    pole_mask = np.abs(phi) > (np.pi * 0.4)
    equator_mask = np.abs(phi) < (np.pi * 0.1)

    pole_density = np.mean(pole_mask)
    equator_density = np.mean(equator_mask)

    print(f"📌 Pole ratio: {pole_density:.4f}")
    print(f"📌 Equator ratio: {equator_density:.4f}")

    # =========================================================
    # ⑥ レポート保存
    # =========================================================
    with open(f"{output_dir}/report.txt", "w") as f:
        f.write("SPaGS Diagnosis\n")
        f.write("================\n")
        f.write(f"mean_r: {r_peak}\n")
        f.write(f"phi_entropy: {entropy}\n")
        f.write(f"spherical_coverage: {coverage}\n")
        f.write(f"high_freq: {hf_mean}\n")
        f.write(f"pole_ratio: {pole_density}\n")
        f.write(f"equator_ratio: {equator_density}\n")

    print("✅ SPaGS診断完了")