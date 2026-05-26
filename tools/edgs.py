import os
import glob
import numpy as np
import healpy as hp
import cv2
import json
import open3d as o3d
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from collections import defaultdict


# =========================
# Pose Loader
# =========================
def load_openmvg_poses(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    view_id_to_filename = {}
    for view in data.get('views', []):
        try:
            v_inner = view['value']['ptr_wrapper']['data']
        except KeyError:
            v_inner = view['value']
        view_id_to_filename[v_inner['id_pose']] = v_inner['filename'].split('.')[0]

    poses = {}
    for ext in data.get('extrinsics', []):
        pose_id = ext['key']
        if pose_id not in view_id_to_filename:
            continue
        fname = view_id_to_filename[pose_id]
        val = ext['value']
        poses[fname] = {
            'R_inv': np.array(val['rotation']).T,
            'center': np.array(val['center']).flatten()
        }

    print(f"✅ Loaded {len(poses)} camera poses.")
    return poses


# =========================
# Adaptive Sampling
# =========================
def get_adaptive_nside(d):
    if d < 2.0: return 128
    if d < 5.0: return 64
    if d < 10.0: return 32
    return 16


# =========================
# Depth Clustering
# =========================
def cluster_by_depth(points, max_k=3, depth_thresh=0.2):
    if len(points) == 0:
        return []

    points = sorted(points, key=lambda x: x['depth'])
    clusters = [[points[0]]]

    for p in points[1:]:
        if abs(p['depth'] - clusters[-1][-1]['depth']) < depth_thresh:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    return clusters[:max_k]


# =========================
# Main Function
# =========================
def generate_unified_edgs_init(
    existing_ply, depth_dir, rgb_dir, mvg_json, output_ply,
    reduction_ratio=0.3,
    fine_tune_factor=1.5,
    max_per_anchor=3,
    depth_thresh=0.2
):
    print("🔄 Loading SfM points...")
    old_ply = PlyData.read(existing_ply)
    v = old_ply['vertex']

    full_xyz = np.stack([v['x'], v['y'], v['z']], axis=1)
    full_rgb = np.stack([v['red'], v['green'], v['blue']], axis=1)

    sfm_tree = cKDTree(full_xyz)
    exclusion_radius = 0.04

    sfm_center = np.mean(full_xyz, axis=0)
    sfm_median = np.median(np.linalg.norm(full_xyz - sfm_center, axis=1))

    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))

    sample_depths = [
        np.median(cv2.imread(f, cv2.IMREAD_ANYDEPTH)[cv2.imread(f, cv2.IMREAD_ANYDEPTH) > 0])
        for f in depth_files[:5]
    ]
    final_scale = (sfm_median / np.mean(sample_depths)) * fine_tune_factor

    poses = load_openmvg_poses(mvg_json)

    anchors = defaultdict(list)
    target_nsides = [16, 32, 64, 128]

    print("🚀 Generating anchors...")

    for d_path, r_path in zip(depth_files[::2], rgb_files[::2]):
        fname = os.path.basename(d_path).split('.')[0]
        fname_key = ''.join(filter(str.isdigit, fname)).lstrip('0') or '0'
        if fname_key not in poses:
            continue

        p = poses[fname_key]
        R_inv, center = p['R_inv'], p['center']

        depth_img = cv2.imread(d_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) * final_scale
        rgb_img = cv2.cvtColor(cv2.imread(r_path), cv2.COLOR_BGR2RGB)

        h, w = depth_img.shape

        for nside in target_nsides:
            npix = hp.nside2npix(nside)
            ang_res = np.sqrt((4 * np.pi) / npix)
            hp_vectors = np.array(hp.pix2vec(nside, np.arange(npix))).T

            for pix_idx, v_cam in enumerate(hp_vectors):

                v_world = R_inv @ v_cam

                v_mod = v_cam[[2, 0, 1]]
                theta = np.arctan2(v_mod[1], v_mod[0])
                phi = np.arcsin(np.clip(v_mod[2], -1.0, 1.0))

                u = int((theta + np.pi) / (2 * np.pi) * w) % w
                v_ = int((phi + np.pi / 2) / np.pi * h) % h

                d = depth_img[v_, u]
                if d <= (0.05 * final_scale):
                    continue

                if get_adaptive_nside(d / final_scale) != nside:
                    continue

                pos = (v_world * d) + center

                dist, _ = sfm_tree.query(pos, distance_upper_bound=exclusion_radius)
                if dist < exclusion_radius:
                    continue

                lat_s = d * ang_res * 0.6
                dep_s = d * 0.001

                anchors[(nside, pix_idx)].append({
                    'pos': pos,
                    'depth': d,
                    'color': rgb_img[v_, u],
                    'lat_s': lat_s,
                    'dep_s': dep_s
                })

    print("🧩 Aggregating anchors...")

    temp_points = []
    temp_attr = []

    anchor_sizes = []
    cluster_counts = []

    for key, pts in anchors.items():
        anchor_sizes.append(len(pts))

        clusters = cluster_by_depth(pts, max_k=max_per_anchor, depth_thresh=depth_thresh)
        cluster_counts.append(len(clusters))

        for cluster in clusters:
            pos = np.mean([p['pos'] for p in cluster], axis=0)
            col = np.mean([p['color'] for p in cluster], axis=0)
            lat_s = np.mean([p['lat_s'] for p in cluster])
            dep_s = np.mean([p['dep_s'] for p in cluster])

            importance = len(cluster)  # 🔥重要

            temp_points.append(pos)
            temp_attr.append({
                'col': col.astype(np.uint8),
                's': [dep_s, lat_s, lat_s],
                'imp': importance
            })

    print(f"🔢 Reduced to {len(temp_points)} Gaussians")

    print("🎯 Running ICP...")
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(np.array(temp_points))

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(full_xyz)

    reg = o3d.pipelines.registration.registration_icp(source, target, 0.2, np.eye(4))
    source.transform(reg.transformation)

    aligned_xyz = np.asarray(source.points)

    print("💾 Saving...")

    all_elements = []

    num_reduced = min(len(full_xyz), int(len(full_xyz) * reduction_ratio))
    idx = np.random.choice(len(full_xyz), num_reduced, replace=False)

    # SfM点
    for i in idx:
        all_elements.append((
            full_xyz[i, 0], full_xyz[i, 1], full_xyz[i, 2],
            full_rgb[i][0], full_rgb[i][1], full_rgb[i][2],
            0.005, 0.005, 0.005,
            1.0,
            1.0
        ))

    # 生成ガウス
    for i, pos in enumerate(aligned_xyz):
        c = temp_attr[i]['col']
        s = temp_attr[i]['s']
        imp = temp_attr[i]['imp']

        all_elements.append((
            pos[0], pos[1], pos[2],
            c[0], c[1], c[2],
            s[0], s[1], s[2],
            0.01,
            float(imp)
        ))

    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('opacity', 'f4'),
        ('importance', 'f4')
    ]

    el = PlyElement.describe(np.array(all_elements, dtype=vertex_dtype), 'vertex')
    PlyData([el]).write(output_ply)

    print("\n📊 ===== 統計情報 =====")
    print(f"アンカー総数　　　: {len(anchors)}")
    print(f"生成ガウス数　　　: {len(temp_points)}")
    print(f"SfM点数　　　　　: {num_reduced}")
    print(f"合計点数　　　　　: {num_reduced + len(temp_points)}")
    print(f"ICP移動量　　　　: {reg.transformation[:3, 3]}")
    print("======================\n")

    print(f"🏁 Saved: {output_ply}")


# =========================
# Run
# =========================
if __name__ == "__main__":
    generate_unified_edgs_init(
        existing_ply='/home/bandailab/nerficg/dataset/OmniBlender/barbershop/openMVG/reconstruction/colorized.ply',
        depth_dir='/home/bandailab/nerficg/output/SPaGS/barbershop_base/test_30000/depth',
        rgb_dir='/home/bandailab/nerficg/output/SPaGS/barbershop_base/test_30000/rgb_gt',
        mvg_json='/home/bandailab/nerficg/dataset/OmniBlender/barbershop/openMVG/data_openmvg.json',
        output_ply='anchor_gaussian_barbershop.ply'
    )