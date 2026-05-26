import numpy as np
import healpy as hp
from plyfile import PlyData, PlyElement
from scipy.interpolate import NearestNDInterpolator

def augment_ply_with_adaptive_healpix(input_ply, output_ply, nside=32):
    # 1. 既存のPLYの読み込み
    plydata = PlyData.read(input_ply)
    v = plydata['vertex']
    existing_xyz = np.stack([v['x'], v['y'], v['z']], axis=1)
    existing_rgb = np.stack([v['red'], v['green'], v['blue']], axis=1)

    # 2. 既存の点を方向（HEALPixピクセル）と距離に分解
    dists = np.linalg.norm(existing_xyz, axis=1)
    # 距離が0の点を除外（エラー回避）
    valid = dists > 1e-5
    existing_xyz = existing_xyz[valid]
    existing_rgb = existing_rgb[valid]
    dists = dists[valid]
    
    norms = existing_xyz / dists[:, np.newaxis]
    occupied_pix = hp.vec2pix(nside, norms[:,0], norms[:,1], norms[:,2])

    # 3. 各ピクセルごとの「壁までの距離」を計算
    npix = hp.nside2npix(nside)
    pixel_dist_map = np.full(npix, -1.0)
    
    # 既存点があるピクセルには、その中央値を距離として登録
    unique_pix = np.unique(occupied_pix)
    for p in unique_occupied:
        pixel_dist_map[p] = np.median(dists[occupied_pix == p])

    # 4. 空白ピクセルの距離を、近隣の既存ピクセルから補間（部屋の形を推測）
    known_pix = np.where(pixel_dist_map > 0)[0]
    unknown_pix = np.where(pixel_dist_map <= 0)[0]
    
    # HEALPixのピクセル中心ベクトルを使って近傍補間
    hp_vectors = np.array(hp.pix2vec(nside, np.arange(npix))).T
    interp = NearestNDInterpolator(hp_vectors[known_pix], pixel_dist_map[known_pix])
    pixel_dist_map[unknown_pix] = interp(hp_vectors[unknown_pix])

    # 5. 空白地帯（OpenMVGが点を打てなかった場所）のみに、推定距離で点を追加
    new_xyz = hp_vectors[unknown_pix] * pixel_dist_map[unknown_pix][:, np.newaxis]
    
    # 補完したことがわかるように色を変える（例：赤っぽくする）
    new_rgb = np.full((len(new_xyz), 3), [200, 50, 50], dtype=np.uint8) 

    # 6. 結合と保存
    combined_xyz = np.concatenate([existing_xyz, new_xyz], axis=0)
    combined_rgb = np.concatenate([existing_rgb, new_rgb], axis=0)

    vertex_data = [
        (combined_xyz[i, 0], combined_xyz[i, 1], combined_xyz[i, 2],
         combined_rgb[i, 0], combined_rgb[i, 1], combined_rgb[i, 2])
        for i in range(len(combined_xyz))
    ]
    
    el = PlyElement.describe(
        np.array(vertex_data, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                                     ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]),
        'vertex'
    )
    PlyData([el]).write(output_ply)
    
    print(f"元々の点数: {len(existing_xyz)}")
    print(f"補完された点数: {len(new_xyz)}")
    print(f"保存完了: {output_ply}")

# 実行
augment_ply_with_adaptive_healpix('/home/bandailab/nerficg/dataset/OmniBlender/barbershop/openMVG/reconstruction/colorized.ply', 'adaptive_healpix.ply', nside=32)