#!/usr/bin/env python3
"""
合成実験：球面均一配置のGaussianに対してequirectangularのピクセルカバレッジを計測し、
1/cos(θ)バイアスを実証する。

全Gaussian同一サイズ・同一opacity→仰角以外の条件を揃えることで
equirectangular幾何の寄与だけを分離する。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── パラメータ ──────────────────────────────
N_GAUSSIANS  = 2000         # 球面上に均一配置するGaussian数
SIGMA        = 0.05         # Gaussianの角度スケール [ラジアン]（固定）
IMG_H        = 512          # equirectangular画像の高さ
IMG_W        = 1024         # equirectangular画像の幅
THRESHOLD    = 0.1          # footprintのカウント閾値（ピーク比）
BINS         = np.linspace(-90, 90, 19)
CENTERS      = 0.5 * (BINS[:-1] + BINS[1:])


def fibonacci_sphere(n):
    """Fibonacci格子で球面上にn点を均一配置。"""
    golden = (1 + np.sqrt(5)) / 2
    i      = np.arange(n)
    theta  = np.arccos(1 - 2 * (i + 0.5) / n)   # 極角 [0, π]
    phi    = 2 * np.pi * i / golden               # 方位角
    # 球面座標→直交座標（y軸が上）
    x = np.sin(theta) * np.cos(phi)
    y = np.cos(theta)                             # 天頂=y+
    z = np.sin(theta) * np.sin(phi)
    return np.stack([x, y, z], axis=1)            # [N, 3]


def count_pixels_per_gaussian(positions, sigma, H, W, threshold):
    """
    各GaussianのEquirectangularフットプリント内ピクセル数を計算する。

    equirectangularでは仰角θのピクセルが「横方向に」1/cos(θ)倍密集するため、
    同じ立体角サイズのGaussianが1/cos(θ)倍多くのピクセルをカバーする。
    """
    N = len(positions)

    # 仰角・方位角
    dist_xz = np.sqrt(positions[:, 0]**2 + positions[:, 2]**2).clip(1e-9)
    elev    = np.arctan2(positions[:, 1], dist_xz)   # [-π/2, π/2]
    azim    = np.arctan2(positions[:, 0], positions[:, 2])  # [-π, π]
    cos_th  = np.cos(elev).clip(0.01)

    # equirectangularでの中心ピクセル座標
    cx = ((azim + np.pi) / (2 * np.pi) * W).astype(int) % W  # [N]
    cy = ((np.pi / 2 - elev) / np.pi * H).astype(int).clip(0, H-1)  # [N]

    # equirectangularでのフットプリントサイズ（ピクセル単位）
    # 垂直方向: σ [rad] → σ * H/π pixels
    # 水平方向: σ/cos(θ) [rad] → σ/cos(θ) * W/(2π) pixels  ← 1/cos(θ)バイアス
    r_v = sigma * H / np.pi                    # [N] 垂直半径（全Gaussian同じ）
    r_h = sigma / cos_th * W / (2 * np.pi)    # [N] 水平半径（1/cos(θ)倍）

    # ピクセルカウント ≈ π × r_h × r_v  （楕円面積）
    pixel_counts = np.pi * r_h * r_v          # [N] ← 1/cos(θ)に比例するはず

    return np.degrees(elev), pixel_counts, r_h, r_v, cos_th


def main():
    # 1. 球面上にGaussianを均一配置
    positions = fibonacci_sphere(N_GAUSSIANS)   # [N, 3]

    # 2. 各Gaussianのピクセルカウントを計算
    elev_deg, pixel_counts, r_h, r_v, cos_th = \
        count_pixels_per_gaussian(positions, SIGMA, IMG_H, IMG_W, THRESHOLD)

    # 3. 仰角ビン平均
    binned = np.full(len(CENTERS), np.nan)
    for i, (lo, hi) in enumerate(zip(BINS[:-1], BINS[1:])):
        m = (elev_deg >= lo) & (elev_deg < hi)
        if m.sum() >= 3:
            binned[i] = pixel_counts[m].mean()

    # 赤道基準正規化
    eq_mask = np.abs(CENTERS) < 15
    eq_mean = np.nanmean(binned[eq_mask])
    binned_norm = binned / eq_mean

    # 理論曲線 1/cos(θ)
    cos_th_centers = np.cos(np.radians(CENTERS)).clip(0.05)
    theory = (1 / cos_th_centers) / np.mean(1 / cos_th_centers[eq_mask])

    # 数値サマリ
    print(f"{'Lat':>8} | {'N Gauss':>8} | {'PixelCount':>10} | {'NormCount':>10} | {'Theory':>8}")
    print("-" * 55)
    for i, lat in enumerate(CENTERS):
        m = (elev_deg >= BINS[i]) & (elev_deg < BINS[i+1])
        if not np.isnan(binned_norm[i]):
            print(f"{lat:>+8.1f}° | {m.sum():>8d} | {binned[i]:>10.1f} | {binned_norm[i]:>10.3f} | {theory[i]:>8.3f}")

    # 4. プロット
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── 左：ピクセルカウント vs 仰角 ──
    ax = axes[0]
    valid = ~np.isnan(binned_norm)
    ax.axvspan(-90, -60, alpha=0.10, color='#4477AA', zorder=0)
    ax.axvspan( 60,  90, alpha=0.10, color='#4477AA', zorder=0)
    ax.plot(CENTERS[valid], binned_norm[valid],
            color='#E05A2B', lw=2.5, marker='o', markersize=5,
            label='Pixel coverage (measured)', zorder=5)
    ax.plot(CENTERS[valid], theory[valid],
            color='#4477AA', lw=2.0, ls='--',
            label='1/cos(θ) theory', zorder=4)
    ax.axhline(1.0, color='gray', lw=1.2, ls=':', label='Equator reference')
    ymax = theory[valid].max() * 1.15
    ax.set_ylim(0, min(ymax, 12))
    ax.text(-75, min(ymax, 12)*0.95, 'Polar\nzone', ha='center', va='top',
            fontsize=9, color='#4477AA', style='italic')
    ax.text( 75, min(ymax, 12)*0.95, 'Polar\nzone', ha='center', va='top',
            fontsize=9, color='#4477AA', style='italic')
    ax.set_xlabel('Gaussian elevation angle [deg]', fontsize=12)
    ax.set_ylabel('Normalized pixel coverage\n(equatorial = 1.0)', fontsize=11)
    ax.set_title(f'Synthetic experiment (N={N_GAUSSIANS} identical Gaussians on sphere)\n'
                 'Equirectangular pixel coverage ∝ 1/cos(θ)', fontsize=11)
    ax.set_xlim(-90, 90)
    ax.set_xticks(range(-90, 91, 15))
    ax.legend(fontsize=10)
    ax.grid(ls='--', alpha=0.4)

    # ── 右：equirectangular可視化（どこにGaussianがあるか）──
    ax = axes[1]
    # equirectangularグリッドにGaussianの位置を点で描画
    dist_xz = np.sqrt(positions[:, 0]**2 + positions[:, 2]**2).clip(1e-9)
    elev_r  = np.arctan2(positions[:, 1], dist_xz)
    azim_r  = np.arctan2(positions[:, 0], positions[:, 2])
    u = (azim_r + np.pi) / (2 * np.pi)   # [0, 1]
    v = (np.pi/2 - elev_r) / np.pi        # [0, 1]  (top=north pole)

    # ピクセルカウントで色付け
    sc = ax.scatter(u * IMG_W, v * IMG_H, c=pixel_counts,
                    cmap='hot_r', s=4, alpha=0.7,
                    vmin=pixel_counts.min(), vmax=np.percentile(pixel_counts, 95))
    plt.colorbar(sc, ax=ax, label='Pixel coverage (pixels)')
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)
    ax.set_xlabel('Azimuth (pixel)', fontsize=11)
    ax.set_ylabel('Elevation (pixel)', fontsize=11)
    ax.set_title('Pixel coverage per Gaussian\nin equirectangular space\n'
                 '(red=more pixels, polar regions over-counted)', fontsize=11)
    # 緯度線
    for lat in [-60, -30, 0, 30, 60]:
        y_px = (np.pi/2 - np.radians(lat)) / np.pi * IMG_H
        ax.axhline(y_px, color='white', lw=0.8, alpha=0.6)
        ax.text(IMG_W*0.02, y_px-5, f'{lat:+d}°', color='white', fontsize=7, va='bottom')

    fig.tight_layout()
    out = Path('results/figures/synthetic_score_bias.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved → {out}')


if __name__ == '__main__':
    main()
