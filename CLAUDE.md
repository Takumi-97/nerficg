# CLAUDE.md — nerficg / SPaGS 360° Pruning Project

## プロジェクト概要

PyTorchベースのNeRF・3DGSフレームワーク（nerficg）上で、**SPaGS（Spherical Panoramic Gaussian Splatting）** を拡張し、**360度equirectangularシーンに特化したGaussianプルーニング手法**を研究・実装している。

## 主な作業ディレクトリ

```
src/Methods/SPaGS/
├── Trainer.py   # 学習ループ・プルーニング呼び出し
├── Renderer.py  # レンダリング・Contribution score計算
├── Model.py     # Gaussiansクラス・各プルーニングメソッド
└── Loss.py      # WS-L1 + DSSIM損失
```

## 実装済み機能

### 提案手法：Contribution-based Pruning
- `computeSphericalContributionScores()` (Renderer.py): 各Gaussianの貢献スコアを複数フレームから計算
  - 体積項 γ(Σ) = V_norm^β（小Gaussianの過剰ペナルティ防止）
  - 仰角補正 f(θ) = cos(θ)（極の距離バイアス除去）
  - HFフレームブースト（GTグラジェントでフレーム重み付け）
  - エラー重み（高誤差領域のGaussianを削減）

### プルーニング戦略（Model.py）
| メソッド | フラグ | 説明 |
|---|---|---|
| `region_normalized_pruning` | `USE_REGION_NORMALIZED=True`（現在のデフォルト） | 3領域内z-score正規化→グローバルtop-K |
| `region_aware_pruning` | - | 近景赤道/近景極/遠景×HF/LFの6サブグループで保持率を変える |
| `opacity_pruning` | `USE_OPACITY_BASELINE=True` | ベースライン：opacity一様プルーニング |
| `uniform_score_pruning` | `USE_UNIFORM_CONTRIBUTION=True` | アブレーション：スコアで一様グローバルプルーニング |

### その他
- **WS-L1 Loss** (Loss.py): sin(π(i+0.5)/H) で緯度重み付け（WS-PSNR最適化）
- **SA_OPACITY_PRUNING**: densification時に極付近のGaussianに高い不透明度閾値を適用
- **LAT_DENSIFY_CORRECTION**: 極ほど多い勾配水増しをcos(θ)で補正
- **Post-pruning LR reset**: プルーニング後にposition LRを一時的にリセット
- **品質ログ**: `quality_metrics_full.csv`（PSNR/WS-PSNR/SSIM/LPIPS）、`pruning_log.csv`
- **iter_pointcloud/**: 1000iter毎にGaussianの点群をPLYで保存

## プルーニングスケジュール

```
Iter 0-15000:  Densification（Split/Duplicate/Prune）
Iter 16000:    1回目 contributionBasedPruning
Iter 20000:    2回目 contributionBasedPruning
Iter 30000:    学習終了
```

## 評価指標
- PSNR, WS-PSNR（球面加重PSNR）, SSIM, LPIPS
- Gaussian数（プルーニング前後の削減率）

## アブレーション設計
`EXPERIMENT_TAG`で実験を区別（"proposed" / "baseline" / "opacity" など）

## 注意点
- equirectangular前提：極付近はピクセルが過密で距離バイアスが生じる
- 遠景のHF判定は遠景内ローカルパーセンタイルで行う（グローバル閾値だと全てLF扱い）
- densification中にプルーニングした場合は`reset_densification_info()`が必要（shape mismatch防止）
