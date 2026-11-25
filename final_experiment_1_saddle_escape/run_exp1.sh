#!/bin/bash
# Reproduces Final Experiment 1; about 20 min per seed on an Apple M4 GPU.
# The raw files in raw/ are the original runs; this script writes to raw_rerun/ so they are never overwritten.
set -e
cd "$(dirname "$0")/.."
OUT=final_experiment_1_saddle_escape/raw_rerun
mkdir -p "$OUT"
for s in 0 1 2; do
  g=1; [ $s = 0 ] && g=31   # loss landscapes only for seed 0
  python3 fim_noise_sgd.py --pool avg --init_scale 0.1 --lr 0.01 --rho1 1e-7 --rho2 1e-6 --batch_size 128 \
    --epochs 50 --seed $s --grid_n $g --out "$OUT/E1_smallinit_seed$s.json" > "$OUT/E1_smallinit_seed$s.log" 2>&1
done
echo "done; to plot the rerun, point make_figure_exp1.py at $OUT"
