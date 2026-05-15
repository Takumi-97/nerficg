import json
import numpy as np
import os
from pathlib import Path
from natsort import natsorted

# ★ 確定したデータセットのルートパスを設定
BARBERSHOP_DATASET_PATH = "/home/bandailab/nerficg/dataset/OmniBlender/barbershop"

def rotmat2qvec(R):
    # quaternion_to_R 関数が期待するクォータニオン形式への変換 (OmniBlender.pyで使用されているものに合わせる)
    [[Rxx, Ryx, Rzx], [Rxy, Ryy, Rzy], [Rxz, Ryz, Rzz]] = R
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def load_file_list(file_path):
    """train.txt/test.txtからファイル名リストを読み込む"""
    if not Path(file_path).exists():
        return set()
    with open(file_path, 'r') as f:
        # 各行が '00001' のようなインデックスであることを想定し、'.png' を付けてセットにする
        return {line.strip() + '.png' for line in f if line.strip()}


def convert_blender_to_openmvg(base_dir):
    base_path = Path(base_dir)
    transform_file = base_path / 'transform.json'
    if not transform_file.exists():
        print(f"ERROR: transform.json not found in {base_dir}")
        return

    # 1. transform.json の読み込み
    with open(transform_file, 'r') as f:
        blender_data = json.load(f)

    # 2. train.txt / test.txt の読み込み
    train_names = load_file_list(base_path / 'train.txt')
    test_names = load_file_list(base_path / 'test.txt')
    # valはtestと同じとして処理することが多いが、ここではtrainに含める

    # 3. 共通のイントリンシクス情報を作成
    fx = blender_data.get('fl_x', blender_data.get('focal', 500))
    fy = blender_data.get('fl_y', fx)
    w = blender_data.get('w', 800)
    h = blender_data.get('h', 800)
    cx = w / 2.0
    cy = h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).tolist()
    
    intrinsics_data = [{
        "value": {"ptr_wrapper": {"data": {"value0": {"width": w, "height": h, "K": K}}}}
    }]

    # 4. データセットを分割して格納
    output_data_train = {"intrinsics": intrinsics_data, "extrinsics": [], "views": []}
    output_data_test = {"intrinsics": intrinsics_data, "extrinsics": [], "views": []}
    
    current_train_idx = 0
    current_test_idx = 0

    for frame in blender_data["frames"]:
        
        # Blenderのファイルパスから画像ファイル名（例: 00001.png）を抽出
        image_name = Path(frame["file_path"]).stem + '.png' 
        
        # 姿勢情報 (c2w) を抽出
        c2w_matrix = np.array(frame['transform_matrix']).astype(np.float32)
        R_openmvg = c2w_matrix[:3, :3].T 
        C_center = -R_openmvg @ c2w_matrix[:3, 3] 
        
        extrinsic = {"id_pose": -1, "rotation": R_openmvg.tolist(), "center": C_center.tolist()}
        view = {"ptr_wrapper": {"data": {"id_pose": -1, "filename": image_name}}}

        target_list = None
        target_idx = -1
        
        # ファイル名に基づいて train/test を判定
        if image_name in train_names:
            target_list = output_data_train
            target_idx = current_train_idx
            current_train_idx += 1
        elif image_name in test_names:
            target_list = output_data_test
            target_idx = current_test_idx
            current_test_idx += 1
        
        if target_list is not None:
            # ExtrinsicsとViewのIDを連番に設定
            extrinsic["id_pose"] = target_idx
            view["ptr_wrapper"]["data"]["id_pose"] = target_idx
            
            target_list["extrinsics"].append({"value": extrinsic})
            target_list["views"].append({"value": view})


    # 5. 最終的なJSONファイルを保存
    output_dir = base_path / 'openMVG'
    output_dir.mkdir(exist_ok=True) 
    
    with open(output_dir / 'data_openmvg_train.json', 'w') as f:
        json.dump(output_data_train, f, indent=4)
        
    with open(output_dir / 'data_openmvg_test.json', 'w') as f:
        json.dump(output_data_test, f, indent=4)
        
    # valデータがない場合、trainのコピーをvalとして作成（ローダーがvalも探すため）
    if not (output_dir / 'data_openmvg_val.json').exists():
         with open(output_dir / 'data_openmvg_val.json', 'w') as f:
            json.dump(output_data_train, f, indent=4)
            
    print(f"✅ Conversion successful!")
    print(f"   -> Train views: {len(output_data_train['views'])}")
    print(f"   -> Test views: {len(output_data_test['views'])}")


if __name__ == '__main__':
    print(f"Attempting to convert data in: {BARBERSHOP_DATASET_PATH}")
    convert_blender_to_openmvg(BARBERSHOP_DATASET_PATH)
