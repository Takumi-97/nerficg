#!/usr/bin/env python3
"""
run_experiments.py
==================
複数の実験条件 × 複数シーンを自動実行し、結果をCSVにまとめるスクリプト。

セットアップ:
    このスクリプトを ~/nerficg/ に置いて実行する。

使い方:
    # 全条件を barbershop で実行
    python run_experiments.py --scenes barbershop

    # 複数シーン × 全条件
    python run_experiments.py --scenes barbershop archiviz-flat gs_garden

    # 特定の条件のみ
    python run_experiments.py --scenes barbershop --conditions baseline contribution_60

    # ドライラン（設定確認のみ、実行しない）
    python run_experiments.py --scenes barbershop --dry-run

    # 既存の結果をまとめるだけ
    python run_experiments.py --summarize-only --scenes barbershop

    # 利用可能な条件を確認
    python run_experiments.py --list-conditions

シーンとデータパスの対応:
    デフォルトでは dataset/OmniBlender/<scene_name> を使用。
    別のパスの場合は --dataset-root で指定。
    例: --dataset-root dataset/OmniBlender
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ============================================================
# 実験条件の定義
# ============================================================
CONDITIONS = {
    "baseline": {
        "desc": "Pruningなし（ベースライン）",
        "training_overrides": {
            "USE_VISIBILITY_PRUNING": "false",
            "USE_CONTRIBUTION_PRUNING": "false",
            "USE_HF_DENSIFICATION": "false",
            "CONTRIBUTION_PRUNING_KEEP_RATIO": "0.0",
            "CONTRIBUTION_PRUNING_ITERATIONS": "[16000, 24000]",
            "CONTRIBUTION_PRUNING_USE_VOLUME": "false",
            "CONTRIBUTION_PRUNING_USE_SPHERICAL": "false",
            "CONTRIBUTION_PRUNING_BETA": "0.5",
            "CONTRIBUTION_PRUNING_SAMPLE_FRAMES": "200",
            "USE_HF_DENSIFICATION": "false",
        },
        "loss_overrides": {},
    },
    "opacity_pruning": {
        "desc": "SPaGS純正 Opacity Pruning (threshold=0.10)",
        "training_overrides": {
            "USE_VISIBILITY_PRUNING": "true",
            "VISIBILITY_PRUNING_THRESHOLD": "0.10",
            "USE_CONTRIBUTION_PRUNING": "false",
            "USE_HF_DENSIFICATION": "false",
        },
        "loss_overrides": {},
    },
    "contribution_44": {
        "desc": "提案手法 ~44%削減 (keep=0.75×2回)",
        "training_overrides": {
            "USE_VISIBILITY_PRUNING": "false",
            "USE_CONTRIBUTION_PRUNING": "true",
            "CONTRIBUTION_PRUNING_KEEP_RATIO": "0.75",
            "CONTRIBUTION_PRUNING_ITERATIONS": "[16000, 24000]",
            "CONTRIBUTION_PRUNING_USE_VOLUME": "true",
            "CONTRIBUTION_PRUNING_USE_SPHERICAL": "true",
            "CONTRIBUTION_PRUNING_BETA": "0.5",
            "CONTRIBUTION_PRUNING_SAMPLE_FRAMES": "200",
            "USE_HF_DENSIFICATION": "false",
        },
        "loss_overrides": {},
    },
    "contribution_60": {
        "desc": "提案手法 ~60%削減 (keep=0.632×2回)",
        "training_overrides": {
            "USE_VISIBILITY_PRUNING": "false",
            "USE_CONTRIBUTION_PRUNING": "true",
            "CONTRIBUTION_PRUNING_KEEP_RATIO": "0.632",
            "CONTRIBUTION_PRUNING_ITERATIONS": "[16000, 24000]",
            "CONTRIBUTION_PRUNING_USE_VOLUME": "true",
            "CONTRIBUTION_PRUNING_USE_SPHERICAL": "true",
            "CONTRIBUTION_PRUNING_BETA": "0.5",
            "CONTRIBUTION_PRUNING_SAMPLE_FRAMES": "200",
            "USE_HF_DENSIFICATION": "false",
        },
        "loss_overrides": {},
    },
    "contribution_75": {
        "desc": "提案手法 ~75%削減 (keep=0.5×2回)",
        "training_overrides": {
            "USE_VISIBILITY_PRUNING": "false",
            "USE_CONTRIBUTION_PRUNING": "true",
            "CONTRIBUTION_PRUNING_KEEP_RATIO": "0.5",
            "CONTRIBUTION_PRUNING_ITERATIONS": "[16000, 24000]",
            "CONTRIBUTION_PRUNING_USE_VOLUME": "true",
            "CONTRIBUTION_PRUNING_USE_SPHERICAL": "true",
            "CONTRIBUTION_PRUNING_BETA": "0.5",
            "CONTRIBUTION_PRUNING_SAMPLE_FRAMES": "200",
            "USE_HF_DENSIFICATION": "false",
        },
        "loss_overrides": {},
    },
    # アブレーション: 体積補正のみ
    "ablation_volume_only": {
        "desc": "アブレーション: 体積補正のみ (球面補正なし)",
        "training_overrides": {
            "USE_VISIBILITY_PRUNING": "false",
            "USE_CONTRIBUTION_PRUNING": "true",
            "CONTRIBUTION_PRUNING_KEEP_RATIO": "0.632",
            "CONTRIBUTION_PRUNING_ITERATIONS": "[16000, 24000]",
            "CONTRIBUTION_PRUNING_USE_VOLUME": "true",
            "CONTRIBUTION_PRUNING_USE_SPHERICAL": "false",
            "USE_HF_DENSIFICATION": "false",
        },
        "loss_overrides": {},
    },
    # アブレーション: 球面補正のみ
    "ablation_spherical_only": {
        "desc": "アブレーション: 球面補正のみ (体積補正なし)",
        "training_overrides": {
            "USE_VISIBILITY_PRUNING": "false",
            "USE_CONTRIBUTION_PRUNING": "true",
            "CONTRIBUTION_PRUNING_KEEP_RATIO": "0.632",
            "CONTRIBUTION_PRUNING_ITERATIONS": "[16000, 24000]",
            "CONTRIBUTION_PRUNING_USE_VOLUME": "false",
            "CONTRIBUTION_PRUNING_USE_SPHERICAL": "true",
            "USE_HF_DENSIFICATION": "false",
        },
        "loss_overrides": {},
    },
}

DEFAULT_CONDITIONS = [
    "baseline",
    "opacity_pruning",
    "contribution_44",
    "contribution_60",
    "contribution_75",
]


# ============================================================
# yaml パッチ処理
# ============================================================

def patch_yaml(
    base_yaml: Path,
    scene: str,
    dataset_root: str,
    training_overrides: dict,
    loss_overrides: dict,
    out_yaml: Path,
) -> None:
    """
    ベースyamlを読み込み、以下を書き換えて out_yaml に出力する：
    - TRAINING.MODEL_NAME → "{scene}_{condition}"
    - DATASET.PATH        → "{dataset_root}/{scene}"
    - TRAINING配下の各キー → training_overrides の値
    - TRAINING.LOSS配下    → loss_overrides の値（USE_SPHERICAL_WEIGHTED_L1 等）
    """
    with open(base_yaml) as f:
        lines = f.readlines()

    result = []
    section = None          # 現在のトップレベルセクション
    in_loss = False         # LOSS: サブセクション内か

    for line in lines:
        stripped = line.strip()

        # トップレベルセクション検出（先頭が非スペースで : で終わる）
        if stripped and not line.startswith(" ") and stripped.endswith(":"):
            section = stripped[:-1]
            in_loss = False
            result.append(line)
            continue

        # TRAINING セクション内
        if section == "TRAINING":

            # LOSS: サブセクション検出
            if stripped == "LOSS:":
                in_loss = True
                result.append(line)
                continue

            # LOSS: サブセクション内
            if in_loss:
                if stripped and not line.startswith("    "):
                    # LOSSセクション終了
                    in_loss = False
                else:
                    # loss_overrides のキーを上書き
                    for key, val in loss_overrides.items():
                        if stripped.startswith(f"{key}:"):
                            indent = len(line) - len(line.lstrip())
                            result.append(" " * indent + f"{key}: {val}\n")
                            break
                    else:
                        result.append(line)
                    continue

            # MODEL_NAME を上書き
            if stripped.startswith("MODEL_NAME:"):
                result.append(f"  MODEL_NAME: {scene}\n")
                continue

            # training_overrides のキーを上書き
            matched = False
            for key, val in training_overrides.items():
                if stripped.startswith(f"{key}:"):
                    indent = len(line) - len(line.lstrip())
                    result.append(" " * indent + f"{key}: {val}\n")
                    matched = True
                    break
            if matched:
                continue

        # DATASET セクション内
        if section == "DATASET":
            if stripped.startswith("PATH:"):
                result.append(f"  PATH: {dataset_root}/{scene}\n")
                continue

        result.append(line)

    with open(out_yaml, "w") as f:
        f.writelines(result)


# ============================================================
# 結果抽出
# ============================================================

def extract_final_metrics(log_file: Path) -> dict:
    """quality_metrics_full.csv の最終行からメトリクスを取得する。"""
    if not log_file.exists():
        return {}
    rows = []
    with open(log_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {}
    last = rows[-1]
    result = {
        "psnr":   float(last["psnr"])   if last.get("psnr")   else None,
        "ssim":   float(last["ssim"])   if last.get("ssim")   else None,
        "lpips":  float(last["lpips"])  if last.get("lpips")  else None,
        "points": int(last["points"])   if last.get("points") else None,
        "ws_psnr":float(last["ws_psnr"])if last.get("ws_psnr") else None,
    }
    return result


def extract_pruning_info(pruning_log: Path) -> dict:
    """pruning_log.csv から削減率情報を取得する。"""
    if not pruning_log.exists():
        return {}
    rows = []
    with open(pruning_log) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {}
    n_before = int(rows[0]["n_before"])
    n_after  = int(rows[-1]["n_after"])
    total_pct = (n_before - n_after) / n_before * 100 if n_before > 0 else 0
    return {
        "n_before":             n_before,
        "n_after":              n_after,
        "total_reduction_pct":  round(total_pct, 1),
        "n_pruning_events":     len(rows),
    }


# ============================================================
# 実験実行
# ============================================================

def run_one_experiment(
    scene: str,
    condition: str,
    base_yaml: Path,
    dataset_root: str,
    output_dir: Path,
    gpu: int,
    dry_run: bool,
) -> dict:
    """1実験を実行して結果dictを返す。"""
    cond = CONDITIONS[condition]
    exp_tag  = f"{scene}__{condition}"
    exp_dir  = output_dir / exp_tag
    exp_dir.mkdir(parents=True, exist_ok=True)

    # パッチ済みyaml生成
    patched_yaml = exp_dir / "config.yaml"
    patch_yaml(
        base_yaml       = base_yaml,
        scene           = scene,
        dataset_root    = dataset_root,
        training_overrides = cond["training_overrides"],
        loss_overrides  = cond.get("loss_overrides", {}),
        out_yaml        = patched_yaml,
    )

    print(f"\n{'='*65}")
    print(f"  [{exp_tag}]")
    print(f"  {cond['desc']}")
    print(f"{'='*65}")

    if dry_run:
        print(f"  設定ファイル: {patched_yaml}")
        print(f"  [DRY RUN] スキップ")
        return {"scene": scene, "condition": condition, "desc": cond["desc"], "status": "dry_run"}

    # カレントディレクトリの既存ログをクリア（前の実験の残骸を防ぐ）
    for fname in ["quality_metrics_full.csv", "pruning_log.csv"]:
        p = Path(fname)
        if p.exists():
            p.unlink()

    start = time.time()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    proc = subprocess.run(
        [sys.executable, "scripts/train.py", "-c", str(patched_yaml)],
        env=env,
    )
    elapsed = time.time() - start

    status = "done" if proc.returncode == 0 else "failed"
    if status == "failed":
        print(f"  [ERROR] 実験失敗 (return code {proc.returncode})")

    # 結果ファイルを exp_dir に保存
    metrics, pruning = {}, {}
    for fname in ["quality_metrics_full.csv", "pruning_log.csv"]:
        src = Path(fname)
        if src.exists():
            dst = exp_dir / fname
            shutil.copy2(src, dst)

    if status == "done":
        metrics = extract_final_metrics(exp_dir / "quality_metrics_full.csv")
        pruning = extract_pruning_info(exp_dir / "pruning_log.csv")

    result = {
        "scene":      scene,
        "condition":  condition,
        "desc":       cond["desc"],
        "status":     status,
        "elapsed_min": round(elapsed / 60, 1),
        **metrics,
        **pruning,
    }

    if status == "done":
        print(f"  完了: {elapsed/60:.1f}分")
        print(f"  PSNR={metrics.get('psnr','—')}  WS-PSNR={metrics.get('ws_psnr','—')}  "
              f"SSIM={metrics.get('ssim','—')}  LPIPS={metrics.get('lpips','—')}")
        if pruning.get("total_reduction_pct") is not None:
            print(f"  削減率={pruning['total_reduction_pct']}%  "
                  f"({pruning.get('n_before','—'):,} → {pruning.get('n_after','—'):,})")

    return result


# ============================================================
# サマリー出力
# ============================================================

SUMMARY_FIELDS = [
    "scene", "condition", "desc", "status", "elapsed_min",
    "points", "n_before", "n_after", "total_reduction_pct",
    "psnr", "ws_psnr", "ssim", "lpips",
]

def save_summary(results: list[dict], csv_path: Path) -> None:
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def print_summary(results: list[dict]) -> None:
    done = [r for r in results if r.get("status") == "done"]
    if not done:
        print("  完了した実験がありません。")
        return

    print(f"\n{'='*90}")
    print(f"  {'シーン':<15} {'条件':<25} {'Gaussian':>9} {'削減率':>7} "
          f"{'PSNR':>7} {'WS-PSNR':>9} {'SSIM':>7} {'LPIPS':>7}")
    print(f"  {'-'*85}")

    prev_scene = None
    for r in done:
        if r["scene"] != prev_scene:
            if prev_scene is not None:
                print(f"  {'':85}")
            prev_scene = r["scene"]

        pts   = f"{r.get('points',0):,}"        if r.get("points")                else "—"
        red   = f"{r.get('total_reduction_pct')}%" if r.get("total_reduction_pct") is not None else "—"
        psnr  = f"{r.get('psnr'):.2f}"          if r.get("psnr")                  else "—"
        wpsnr = f"{r.get('ws_psnr'):.2f}"       if r.get("ws_psnr")               else "—"
        ssim  = f"{r.get('ssim'):.4f}"          if r.get("ssim")                  else "—"
        lpips = f"{r.get('lpips'):.4f}"         if r.get("lpips")                 else "—"

        print(f"  {r['scene']:<15} {r['condition']:<25} {pts:>9} {red:>7} "
              f"{psnr:>7} {wpsnr:>9} {ssim:>7} {lpips:>7}")

    print(f"{'='*90}")


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SPaGS 実験自動化スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--scenes", nargs="+", required=True,
                        help="シーン名（configs/<scene>.yaml と dataset/OmniBlender/<scene> が存在すること）")
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS,
                        choices=list(CONDITIONS.keys()),
                        help=f"実験条件（デフォルト: {' '.join(DEFAULT_CONDITIONS)}）")
    parser.add_argument("--base-config", default=None,
                        help="ベースyamlのパス。省略時は configs/<scene名>.yaml を使用")
    parser.add_argument("--dataset-root", default="dataset/OmniBlender",
                        help="データセットのルートパス（デフォルト: dataset/OmniBlender）")
    parser.add_argument("--output-dir", default="experiment_results",
                        help="実験結果の保存先（デフォルト: experiment_results/）")
    parser.add_argument("--gpu", type=int, default=0, help="使用するGPU番号")
    parser.add_argument("--dry-run", action="store_true",
                        help="設定確認のみ。実際の訓練は行わない")
    parser.add_argument("--summarize-only", action="store_true",
                        help="既存の結果ファイルをまとめるだけ")
    parser.add_argument("--list-conditions", action="store_true",
                        help="利用可能な実験条件を表示して終了")
    args = parser.parse_args()

    if args.list_conditions:
        print("\n利用可能な実験条件：")
        for k, v in CONDITIONS.items():
            default_mark = " *" if k in DEFAULT_CONDITIONS else ""
            print(f"  {k:<30} {v['desc']}{default_mark}")
        print("\n  * = デフォルト条件")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "summary.csv"

    # summarize-only モード
    if args.summarize_only:
        results = []
        for scene in args.scenes:
            for condition in args.conditions:
                exp_dir = output_dir / f"{scene}__{condition}"
                m = extract_final_metrics(exp_dir / "quality_metrics_full.csv")
                p = extract_pruning_info(exp_dir / "pruning_log.csv")
                if m:
                    results.append({
                        "scene": scene, "condition": condition,
                        "desc": CONDITIONS.get(condition, {}).get("desc", ""),
                        "status": "done", **m, **p,
                    })
        save_summary(results, summary_csv)
        print_summary(results)
        print(f"\n→ サマリーCSV: {summary_csv}")
        return

    # 実験実行モード
    total = len(args.scenes) * len(args.conditions)
    print(f"\n実験計画: {len(args.scenes)} シーン × {len(args.conditions)} 条件 = {total} 実験")
    for scene in args.scenes:
        for condition in args.conditions:
            print(f"  {scene} × {condition}")

    results = []
    done_count = 0

    for scene in args.scenes:
        # ベースyaml: 引数指定 or configs/<scene>.yaml
        if args.base_config:
            base_yaml = Path(args.base_config)
        else:
            base_yaml = Path("configs") / f"{scene}.yaml"

        if not base_yaml.exists():
            print(f"\n[SKIP] yamlが見つかりません: {base_yaml}")
            print(f"       --base-config でパスを指定してください")
            continue

        for condition in args.conditions:
            done_count += 1
            print(f"\n[{done_count}/{total}] {scene} × {condition}")

            result = run_one_experiment(
                scene        = scene,
                condition    = condition,
                base_yaml    = base_yaml,
                dataset_root = args.dataset_root,
                output_dir   = output_dir,
                gpu          = args.gpu,
                dry_run      = args.dry_run,
            )
            results.append(result)

            # 途中経過を随時保存
            save_summary(results, summary_csv)

    print(f"\n{'='*65}")
    print(f"  全実験完了 ({done_count}/{total})")
    print_summary(results)
    save_summary(results, summary_csv)
    print(f"\n→ サマリーCSV: {summary_csv}")


if __name__ == "__main__":
    main()
