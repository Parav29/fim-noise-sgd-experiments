# New experiments for the "Experiments" section — overview

Paper: Ramaswamy, Solanki, Borkar, *Polynomial-Time Handovers from Strict and Degenerate Saddles to Local PL Basins in Deep Learning* (the updated draft dated 24 Nov 2026). Page numbers below are PDF pages of that draft.

Everything here tests only statements that are in that draft. Nothing from outside the paper (toy functions, deep linear networks, uniform-noise comparisons) was added.

| Doc | Experiment | Paper statement it relates to | Outcome (3 seeds each) |
|---|---|---|---|
| `experiment_E1.md` | SGD vs Eq. (1) started near θ = 0 | Contribution 1 (p.2–3); Thm 3 (p.12); remark on p.10 | SGD stays at ln 10 for all 19,500 iterations; Eq. (1) escapes at iteration 1,500 on every seed |
| `experiment_E2.md` | SGD vs Eq. (1) from default init | Contribution 4 (p.3) | Curves coincide (max gap 0.0013); Eq. (1) does no harm |
| `experiment_E3.md` | Escape time vs noise level ρ₁ | Thms 1, 3 (p.7, p.12); noise-to-step-size remarks (p.4, Eq. 35, p.15) | Escape time falls monotonically with ρ₁: 13.3k → 5.0k → 1.5k → 0.53k iterations; measured exponent −0.95 |
| `experiment_E4.md` | No slide-back, constant vs Robbins–Monro step | Thm 5 (p.15), Thm 6 (p.16) | 0 / 6 runs ever return to the plateau |
| `experiment_E5.md` | Loss decay after escape | Linear (PL) vs sublinear (KL) statements (p.2, p.4, p.14) | Rate keeps decaying, then flattens; **not conclusive** about PL vs KL |

Figures: `fig_E1_…png`, `fig_E2_…png`, `fig_E3_…png`, `fig_E4_…png`, `fig_E5_…png`, and `fig_E1_appendix_exp2style_landscape_seed0.png` (the exp2.png-style four-panel figure with loss landscapes, re-made with the new network). Numbers: `summary_numbers.json`. Raw curves and logs: `raw/`.

## Common setup (all experiments)

| Item | Value |
|---|---|
| Data | CIFAR-10 training set, 50,000 images, per-channel normalisation, no augmentation |
| Network | 3 × [Conv 3×3 → GeLU → **AvgPool 2**] with 32, 64, 64 channels → Linear(1024→10); 66,570 parameters; no BatchNorm, no dropout |
| Loss | Cross-entropy |
| Training | batch 128, 50 epochs = 19,500 iterations (last partial batch dropped), no momentum, no weight decay |
| Learning rate | η = 0.01 (constant, except E4's decaying-step runs) |
| Eq. (1) | θ ← θ − α_n (∇L_n + Υ_n), Υ_n^i = s_i ρ₁/(ρ₂ + (∂L_n/∂θ_i)²), s_i = ±1 with prob. ½, drawn independently per coordinate and per step. ρ₂ = 10⁻⁶ everywhere |
| Seeds | 0, 1, 2. The seed fixes the initial weights, the mini-batch order and the sign noise. SGD and Eq. (1) share initial weights and mini-batch order; the signs come from a separate generator |
| Loss recorded | Full 50,000-image training loss every 100 iterations (all analysis uses this), plus the mini-batch loss at every iteration |
| Hardware | Apple M4 GPU (MPS) for all 18 run files, so no cross-device mixing |
| Code | `fim_noise_sgd.py` (flags `--pool avg`, `--step rm`, `--methods`), `run_new_experiments.sh` (all runs), `analyze_new.py` (figures and numbers) |

### Change from the earlier experiments: average pooling
The earlier runs (`exp1.png`, `exp2.png`) used max-pooling. Max-pooling is not smooth, and the paper's Assumption 1 (p.3) requires a C^∞ network. All new runs use average pooling (linear, hence smooth), with GeLU, so the network satisfies Assumption 1. This makes the network train more slowly at η = 0.01 (final loss about 0.95 instead of 0.59 for the small-init run), but all comparisons are between methods on the same network.

### Definitions fixed in code before looking at results
- **Plateau:** the loss of a uniform guess, ln 10 = 2.3026.
- **Escape iteration τ:** the first evaluated iteration (evaluations are every 100 iterations) at which the full training loss is below ln 10 − 0.1 = 2.2026. It is therefore known to ±100 iterations.
- **Slide-back:** after τ, the full training loss is at or above 2.2026 at any later evaluation.
- **L_ref = 0** wherever a log gap is plotted. The true minimum loss L\* is unknown, so "ln L" plots are not gaps to L\*.

## What these experiments do NOT show (applies to all of them)
1. **No test of the PL/KL membership claims (Thms 2, 4, Cor. 1).** They need L\*, the order p of the saddle, and the constants a, b, K_p, none of which can be measured for a CNN.
2. **No test of the escape-time-vs-p statements.** p is not measured or varied. The draft states the p-dependence inconsistently (see below).
3. **The FIM scaling is not isolated.** At the small initialisation, 99.96 % of the coordinates have g² ≪ ρ₂ (measured on one batch, seed 0), so Υ^i ≈ ±ρ₁/ρ₂ for almost every coordinate. The noise is effectively uniform there. This matches the paper's own sentence on p.10 ("isotropic noise initially dominates"), but it means a uniform-noise baseline would probably behave the same, and a reviewer may ask for one.
4. **A competing explanation is untested.** With ηρ₁/ρ₂ = 10⁻³ per coordinate per step, the injected random walk could simply grow the small weights back to ordinary size, after which plain SGD-like training proceeds. We did not log ‖θ_n‖.
5. **The start is near θ = 0, not exactly at it, and θ = 0 being a degenerate saddle of the paper's type is argued (hidden signals vanish), not verified.** On one batch the gradient norm at the start is 0.089, with median coordinate gradient 3·10⁻⁷, so it is concentrated in a few coordinates. We did not break it down by layer.
6. **Retention results (Thm 5) are limits** (η → 0, time exp(ΔV/η)); we can only report absence of slide-back within 19,500 iterations.

## Inconsistencies in the draft that affect how the Experiments section should be worded
- Escape time: Thms 1 and 3 give O(η⁻²(r² + K)); the text after Thm 3 gives r = O(η^(−1/p)), while Eq. (19) gives r ∝ η^(1/p); the text after Thm 5 gives E[τ] = O(η^(−1/(p−1))). The Experiments section should only cite the theorem statements.
- Below Eq. (11) (p.7), ρ₃ is described as a bound on each component of Υ; for the stated lower bound ρ₁/(ρ₂ + ρ₃) it must be a bound on (∂L/∂θ_i)².
- "FIM-scaled": the factor uses the squared mini-batch gradient, not the per-sample Fisher diagonal. The code follows Eq. (1) exactly as written.
