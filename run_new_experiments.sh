#!/bin/bash
# All runs for "new experiments result/". Avg pooling everywhere (paper Assumption 1: C^inf network).
set -e
cd "$(dirname "$0")"
R="new experiments result/raw"
run() { out="$R/$1.json"; shift; [ -f "$out" ] && { echo "skip $out"; return; }
        python3 fim_noise_sgd.py --pool avg "$@" --out "$out" > "${out%.json}.log" 2>&1; echo "done $out"; }

for s in 0 1 2; do
  g=1; [ $s = 0 ] && g=31   # loss-landscape panels only for seed 0 (appendix figure, like exp2.png)
  # E1: start near the stationary point theta=0 (init x0.1), SGD vs Eq. (1), constant step
  run E1_smallinit_seed$s --init_scale 0.1 --rho1 1e-7 --seed $s --grid_n $g
  # E2: default init, SGD vs Eq. (1)
  run E2_defaultinit_seed$s --init_scale 1.0 --rho1 1e-9 --seed $s --grid_n 1
  # E4: Robbins-Monro step size (paper Thm 6), small init
  run E4_rm_seed$s --init_scale 0.1 --rho1 1e-7 --seed $s --grid_n 1 --step rm --rm_n0 2000
  # E3: noise-to-step-size ratio sweep, Eq. (1) only (SGD baseline = E1); rho1=1e-7 is E1 itself
  for r in 1e-8 3e-8 3e-7; do
    run E3_rho1_${r}_seed$s --init_scale 0.1 --rho1 $r --seed $s --grid_n 1 --methods fim_noise
  done
done
echo ALL DONE
