# Experiment E2 — default initialisation: Eq. (1) is harmless

Figure: `fig_E2_defaultinit.png`. Common setup and definitions: `README.md`.

## What it tests
Contribution 4 (p.3) says the framework applies from arbitrary initialisations, not only specially prepared ones. The paper does not claim a speed-up when no saddle slows the optimiser. This experiment checks the weaker, necessary behaviour: that adding Eq. (1)'s noise does not harm training from an ordinary start.

## What exactly is run
- θ₀ = PyTorch default init (scale 1.0), same for both methods.
- A: SGD. B: Eq. (1) with ρ₁ = 10⁻⁹, ρ₂ = 10⁻⁶, η = 0.01 constant (maximum noise ρ₁/ρ₂ = 10⁻³, so at most η ρ₁/ρ₂ = 10⁻⁵ per coordinate per step). ρ₁ was chosen from one measurement of gradient sizes at the starting point (median |g| ≈ 2.7·10⁻⁴, 90th percentile ≈ 9·10⁻⁴), so that ρ₂ is of the order of g² for the larger coordinates.
- 3 seeds, 19,500 iterations.

## Assumptions
- ρ₁ here is a "moderate" value chosen by gradient scale, not tuned. A larger ρ₁ was not tested at default init.

## Results
| Seed | SGD final | Eq. (1) final | Largest gap between the two full-loss curves | Iteration at which each leaves the plateau (loss < 2.2026) |
|---|---|---|---|---|
| 0 | 0.8889 | 0.8899 | 0.0013 | 500 / 500 |
| 1 | 0.9182 | 0.9186 | 0.0007 | 400 / 400 |
| 2 | 0.8821 | 0.8819 | 0.0006 | 600 / 600 |

The two curves coincide to within 0.0013 at every evaluation. The seed-to-seed spread (about 0.04 in the final loss) is far larger than the SGD-vs-Eq. (1) difference.

## What this shows
From a default start both methods behave identically: no plateau worth escaping (SGD itself leaves in 400–600 iterations), and Eq. (1)'s noise at this level neither helps nor hurts. This is consistent with the paper, which makes no speed-up claim for non-saddle starts.

## What this does not show
- It does not show Eq. (1) *enters* a PL/KL region from arbitrary initialisations (Contribution 4's actual content); it only shows no harm at this noise level.
- The noise is so small relative to the gradients that it is essentially inactive here, so "identical" is the expected outcome, not a strong result. A stronger noise level at default init was not run.
