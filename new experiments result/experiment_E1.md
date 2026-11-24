# Experiment E1 — SGD stalls near θ = 0, Eq. (1) escapes

Figures: `fig_E1_escape_smallinit.png` (main), `fig_E1_appendix_exp2style_landscape_seed0.png` (seed 0, with loss landscapes). Common setup and definitions: `README.md`.

## What it tests
The paper's main motivation and claim: plain gradient information cannot escape flat/degenerate saddle regions, whereas FIM-scaled anisotropic noise escapes (abstract; Contribution 1, p.2–3; Thm 3, p.12). The paper also notes (p.10) that near the centre of a flat region isotropic noise dominates the dynamics.

## What exactly is run
- Initial weights θ₀ = 0.1 × (PyTorch default init), same draw for both methods. Starting this small puts θ₀ next to θ = 0, where every hidden activation and back-propagated signal is almost zero (a stationary point of the multi-layer network).
- A: SGD, θ ← θ − η ∇L_n.
- B: Eq. (1), ρ₁ = 10⁻⁷, ρ₂ = 10⁻⁶, constant step η = 0.01 (maximum noise ρ₁/ρ₂ = 0.1 per coordinate; per-step size η ρ₁/ρ₂ = 10⁻³).
- 3 seeds, 19,500 iterations each. The landscape panels (appendix figure) are for seed 0 only: loss on 5,000 fixed images over a 31×31 grid in a filter-normalised random plane around each run's final weights, clipped at 10.

## Assumptions
- Starting from 0.1 × default init is a way of starting near the saddle at θ = 0. This is the standard small-initialisation construction, not the default practice.
- ρ₁ = 10⁻⁷ was chosen in earlier pilot runs on the max-pool network (maximum noise 0.1 escaped, 0.01 did not within 4,000 iterations). The same value was reused here without re-tuning. Standard SGD's learning rate was not tuned for this initialisation.
- Escape is defined as in `README.md` (first evaluated loss < 2.2026).

## Results
| Seed | SGD final loss | SGD: largest \|L − ln 10\| over all evaluations | SGD: lowest mini-batch loss seen | Eq. (1): escape iteration | Eq. (1): final loss |
|---|---|---|---|---|---|
| 0 | 2.30258 | 5.4·10⁻⁶ | 2.3018 | 1,500 | 0.959 |
| 1 | 2.30258 | 7.6·10⁻⁶ | 2.3014 | 1,500 | 0.951 |
| 2 | 2.30258 | 7.4·10⁻⁶ | 2.3017 | 1,500 | 0.944 |

- Standard SGD does not leave the plateau on any seed in 19,500 iterations. The full-set loss changes by less than 10⁻⁵.
- Eq. (1) leaves the plateau at iteration 1,500 on all three seeds and keeps decreasing; the seed-to-seed spread of the whole curve is barely visible in the plot.
- With the earlier max-pool network the same setting escaped at 1,200–1,400 iterations and ended at about 0.59. The qualitative result does not depend on the pooling choice.
- Appendix landscape (seed 0): SGD's final point sits on a flat plateau at 2.303; Eq. (1)'s final point sits at the bottom of a bowl-shaped region. The landscape panels use perturbation directions scaled to each run's own weights, and SGD's weights are still tiny, so the two panels are not on a common scale. They illustrate, not compare.

## What this shows
Near θ = 0, where gradient information is almost zero, plain SGD does not escape within 50 epochs and Eq. (1) does, reproducibly across three seeds. This is the behaviour the paper's escape claim describes.

## What this does not show
- The escape **time** relation (Thm 3 upper bound) is addressed only in E3.
- That the **FIM scaling** is responsible: in this regime the noise is nearly uniform (README, point 3). The experiment supports "the injected noise escapes", not "the g-dependent scaling is what matters".
- That SGD would never escape: only that it does not within 19,500 iterations.
- That the exit point lies in a PL/KL region (README, point 1).
