#!/usr/bin/python
#! -*- encoding: utf-8 -*-

import os
import subprocess
import sys

# --- パス設定 ---
# OpenMVGのバイナリディレクトリ
OPENMVG_SFM_BIN = "/home/bandailab/openMVG_build/Linux-x86_64-RELEASE"
# センサーデータベースのパス
CAMERA_SENSOR_WIDTH_DIRECTORY = "/home/bandailab/openMVG/src/openMVG/exif/sensor_width_database"

# 入力画像ディレクトリ (WSL内の絶対パス)
input_dir = "/home/bandailab/nerficg/dataset/OmniBlender/barbershop/images"
# 出力ディレクトリ (現在のディレクトリに tutorial_out を作成)
output_dir = os.path.abspath("./tutorial_out_360")
matches_dir = os.path.join(output_dir, "matches")
camera_file_params = os.path.join(CAMERA_SENSOR_WIDTH_DIRECTORY, "sensor_width_camera_database.txt")

# ディレクトリ作成
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(matches_dir):
    os.mkdir(matches_dir)

print("Using input dir  : ", input_dir)
print("      output_dir : ", output_dir)

# 1. Intrinsics analysis (360度画像対応)
# -c 7: Spherical camera model (全天球パノラマ) を指定
print("1. Intrinsics analysis (Spherical model)")
pIntrisics = subprocess.Popen([
    os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfMInit_ImageListing"),
    "-i", input_dir,
    "-o", matches_dir,
    "-d", camera_file_params,
    "-c", "7",  # 360度画像 (Spherical) モード
    "-f", "1"
])
pIntrisics.wait()

# 2. Compute features
print("2. Compute features")
pFeatures = subprocess.Popen([
    os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeFeatures"),
    "-i", matches_dir + "/sfm_data.json",
    "-o", matches_dir,
    "-m", "SIFT",
    "-p", "ULTRA",
    "-f", "1"
])
pFeatures.wait()

# 3. Compute matches
print("3. Compute matches")
pMatches = subprocess.Popen([
    os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeMatches"),
    "-i", matches_dir + "/sfm_data.json",
    "-o", matches_dir + "/matches.putative.bin",
    "-f", "1",
    "-n", "AUTO"
])
pMatches.wait()

# 4. Filter matches
print("4. Filter matches")
pFiltering = subprocess.Popen([
    os.path.join(OPENMVG_SFM_BIN, "openMVG_main_GeometricFilter"),
    "-i", matches_dir + "/sfm_data.json",
    "-m", matches_dir + "/matches.putative.bin",
    "-g", "f",
    "-o", matches_dir + "/matches.f.bin"
])
pFiltering.wait()

# 5. Incremental Reconstruction
reconstruction_dir = os.path.join(output_dir, "reconstruction_sequential")
print("5. Do Incremental reconstruction")
pRecons = subprocess.Popen([
    os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfM"),
    "--sfm_engine", "INCREMENTAL",
    "--input_file", matches_dir + "/sfm_data.json",
    "--match_dir", matches_dir,
    "--output_dir", reconstruction_dir
])
pRecons.wait()

# 6. Colorize Structure (点群に色を付ける)
print("6. Colorize Structure")
pColorize = subprocess.Popen([
    os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeSfM_DataColor"),
    "-i", reconstruction_dir + "/sfm_data.bin",
    "-o", os.path.join(reconstruction_dir, "colorized.ply")
])
pColorize.wait()

print("\nFinished! You can check the result in:", os.path.join(reconstruction_dir, "colorized.ply"))