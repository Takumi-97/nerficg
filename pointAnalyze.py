import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData

def analyze_gaussian_distribution(ply_path):
    # 1. PLYファイルの読み込み
    plydata = PlyData.read(ply_path)
    x = np.array(plydata['vertex']['x'])
    y = np.array(plydata['vertex']['y'])
    z = np.array(plydata['vertex']['z'])

    # 2. 3D座標から球面座標 (緯度・経度) への変換
    # 緯度 phi: -pi/2 (南極) から pi/2 (北極)
    # 経度 theta: -pi から pi
    d = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arcsin(z / d)  # 緯度
    theta = np.arctan2(y, x) # 経度

    # 緯度を度数法 (-90° to 90°) に変換
    phi_deg = np.degrees(phi)

    # 3. 可視化: 緯度ごとのヒストグラム
    plt.figure(figsize=(10, 6))
    
    # 理想的な分布（等面積なら cos(phi) に比例するはず）を比較用にプロット
    counts, bins, _ = plt.hist(phi_deg, bins=60, color='skyblue', edgecolor='black', alpha=0.7, label='Actual Distribution')
    
    # 理論上の「面積比」曲線 (赤道付近が多く、極付近が少ないのが正しい)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    ideal_curve = np.cos(np.radians(bin_centers))
    ideal_curve = ideal_curve * (max(counts) / max(ideal_curve)) # スケール合わせ
    plt.plot(bin_centers, ideal_curve, color='red', lw=2, label='Ideal (Equal Area) Trend')

    plt.title('Gaussian Distribution by Latitude')
    plt.xlabel('Latitude (degrees)')
    plt.ylabel('Number of Gaussians')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.show()
    
    # 統計情報の表示
    print(f"Total Gaussians: {len(phi)}")
    print(f"North Pole (>60°): {np.sum(phi_deg > 60)} points")
    print(f"South Pole (<-60°): {np.sum(phi_deg < -60)} points")
    print(f"Equator (-10° to 10°): {np.sum(np.abs(phi_deg) < 10)} points")

# 使用例
analyze_gaussian_distribution('./dataset/OmniBlender/archiviz-flat/openMVG/reconstruction/colorized.ply')