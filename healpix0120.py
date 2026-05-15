import os
import glob
import numpy as np
import healpy as hp
import cv2
from plyfile import PlyData, PlyElement

def get_adaptive_nside(depth_value, is_hierarchical=True, fixed_nside=32):
    """
    is_hierarchical=True  : 距離に応じて 16, 32, 64 を返す
    is_hierarchical=False : 常に fixed_nside を返す
    """
    if not is_hierarchical:
        return fixed_nside
    
    # 提案手法の階層ロジック
    if depth_value < 2.0: return 64   # 近景
    if depth_value < 5.0: return 32   # 中景
    return 16                         # 遠景

def generate_hybrid_ply(existing_ply, depth_dir, rgb_dir, output_ply, 
                        reduction_ratio=0.1, depth_scale=0.001, 
                        is_hierarchical=True, fixed_nside=32):
    """
    is_hierarchical: Trueなら階層的(16/32/64)、Falseなら固定(fixed_nside)
    """
    # 1. 既存点群のロードと間引き
    print(f"既存点群をロード中: {existing_ply}")
    old_ply = PlyData.read(existing_ply)
    v = old_ply['vertex']
    full_xyz = np.stack([v['x'], v['y'], v['z']], axis=1)
    full_rgb = np.stack([v['red'], v['green'], v['blue']], axis=1)

    num_points = len(full_xyz)
    num_reduced = int(num_points * reduction_ratio)
    indices = np.random.choice(num_points, num_reduced, replace=False)
    
    reduced_xyz = full_xyz[indices]
    reduced_rgb = full_rgb[indices]
    print(f"OpenMVG点を間引きました: {num_points} -> {num_reduced} ({reduction_ratio*100}%)")

    # 既存点群の占有判定用 (解像度によらず一貫性を保つため max_nside=64 で判定)
    max_nside = 64
    norm_xyz = reduced_xyz / (np.linalg.norm(reduced_xyz, axis=1, keepdims=True) + 1e-8)
    occupied_pix_64 = set(hp.vec2pix(max_nside, norm_xyz[:,0], norm_xyz[:,1], norm_xyz[:,2]))

    # 2. サンプリング設定
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    
    infill_gaussians = {} # key: (nside, pix_idx)

    # 走査する解像度のリストを決定
    target_nsides = [16, 32, 64] if is_hierarchical else [fixed_nside]
    mode_str = "Hierarchical" if is_hierarchical else f"Flat (NSIDE={fixed_nside})"
    print(f"HEALPix補完開始 [{mode_str}]...")

    # 3. スキャン実行
    for d_path, r_path in zip(depth_files, rgb_files):
        rgb_img = cv2.cvtColor(cv2.imread(r_path), cv2.COLOR_BGR2RGB)
        depth_img = cv2.imread(d_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) * depth_scale
        h, w = depth_img.shape

        for target_nside in target_nsides:
            npix = hp.nside2npix(target_nside)
            hp_vectors = np.array(hp.pix2vec(target_nside, np.arange(npix))).T
            unit_radius = np.sqrt((4 * np.pi) / npix)

            for pix_idx in range(npix):
                if (target_nside, pix_idx) in infill_gaussians: continue

                vec = hp_vectors[pix_idx]
                theta = np.arctan2(vec[1], vec[0])
                phi = np.arcsin(vec[2])
                u = int((theta + np.pi) / (2 * np.pi) * w) % w
                v = int((phi + np.pi/2) / np.pi * h) % h
                
                d = depth_img[v, u]
                if d <= 0.1: continue
                
                # 階層モードなら距離に応じたNSIDEかチェック、固定モードなら常に通過
                if get_adaptive_nside(d, is_hierarchical, fixed_nside) != target_nside: continue

                # 占有チェック
                check_pix_64 = hp.vec2pix(max_nside, vec[0], vec[1], vec[2])
                if check_pix_64 in occupied_pix_64: continue

                # 登録
                infill_gaussians[(target_nside, pix_idx)] = (vec * d, rgb_img[v, u], d * unit_radius)

    # 4. データの結合と保存
    elements = []
    for i in range(len(reduced_xyz)):
        elements.append((reduced_xyz[i,0], reduced_xyz[i,1], reduced_xyz[i,2],
                         reduced_rgb[i,0], reduced_rgb[i,1], reduced_rgb[i,2],
                         0.005, 0.005, 0.005, 1.0))
    for pos, color, s in infill_gaussians.values():
        elements.append((pos[0], pos[1], pos[2], color[0], color[1], color[2],
                         s, s, s, 0.01))

    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                    ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'), ('opacity', 'f4')]
    el = PlyElement.describe(np.array(elements, dtype=vertex_dtype), 'vertex')
    PlyData([el]).write(output_ply)
    
    print(f"完了: {output_ply}")
    print(f" - OpenMVG (10%): {len(reduced_xyz)}, HEALPix: {len(infill_gaussians)}")

# --- 実行例 ---

# 1. 提案手法 (階層あり)
generate_hybrid_ply(
    existing_ply='/home/bandailab/nerficg/dataset/OmniBlender/barbershop/openMVG/reconstruction/colorized_temp.ply',
    depth_dir='/home/bandailab/nerficg/output/SPaGS/barbershop_2026-01-13-16-32-34_SPaGS_Def/test_30000/depth',
    rgb_dir='/home/bandailab/nerficg/output/SPaGS/barbershop_2026-01-13-16-32-34_SPaGS_Def/test_30000/rgb_gt',
    output_ply='colorized_proposed.ply',
    is_hierarchical=True
)

# 2. 比較手法 (固定 NSIDE=32)
generate_hybrid_ply(
    existing_ply='/home/bandailab/nerficg/dataset/OmniBlender/barbershop/openMVG/reconstruction/colorized_temp.ply',
    depth_dir='/home/bandailab/nerficg/output/SPaGS/barbershop_2026-01-13-16-32-34_SPaGS_Def/test_30000/depth',
    rgb_dir='/home/bandailab/nerficg/output/SPaGS/barbershop_2026-01-13-16-32-34_SPaGS_Def/test_30000/rgb_gt',
    output_ply='colorized_flat_32.ply',
    is_hierarchical=False,
    fixed_nside=32
)
