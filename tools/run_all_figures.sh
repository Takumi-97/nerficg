#!/bin/bash
# 全論文図を一括生成する。
# 使用例: bash tools/run_all_figures.sh

set -e
cd "$(dirname "$0")/.."

OUTDIR="results/figures"
mkdir -p "$OUTDIR"

echo "========================================"
echo " 1/5  Pareto curves"
echo "========================================"
python tools/plot_pareto.py --outdir "$OUTDIR"

echo ""
echo "========================================"
echo " 2/5  Latitude band PSNR/SSIM"
echo "========================================"
python tools/plot_latitude.py --outdir "$OUTDIR"

echo ""
echo "========================================"
echo " 3/5  Ablation bar charts"
echo "========================================"
python tools/plot_ablation.py --outdir "$OUTDIR"

echo ""
echo "========================================"
echo " 4/5  Qualitative comparison"
echo "========================================"
python tools/plot_qualitative.py \
    --scenes barbershop lone_monk classroom \
    --frames 0 6 12 \
    --outdir "$OUTDIR"

echo ""
echo "========================================"
echo " 5/5  Gaussian elevation distribution"
echo "========================================"
python tools/plot_gauss_dist.py --outdir "$OUTDIR"

echo ""
echo "========================================"
echo " Done. All figures saved to: $OUTDIR"
echo "========================================"
ls -lh "$OUTDIR"/*.png 2>/dev/null | awk '{print $NF, $5}'
