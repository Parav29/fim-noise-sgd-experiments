# Final Experiment 1 — Noise-assisted escape from a flat stationary region (CIFAR-10)

Figure: `fig_exp1_saddle_escape.png`. Numbers: `summary.json`. Raw data: `raw/E1_smallinit_seed{0,1,2}.json` and `.log`. These are the original training runs (made with the same flags as `run_exp1.sh`).
Reproduce the figure: `python3 make_figure_exp1.py`. Reproduce the training: `./run_exp1.sh` (writes to `raw_rerun/`, never over the originals; about 20 min per seed on an Apple M4 GPU).

## What it shows
Started next to the stationary point θ = 0 of a realistic network, plain SGD does not leave the uniform-guess plateau in 50 epochs. The paper's FIM-noise SGD (Eq. 1) leaves it at iteration 1,500 on every seed and keeps training. This is the paper's motivating claim (abstract; Contribution 1; Thm 3): gradient information alone stalls in flat degenerate regions, and the injected noise escapes.

## Setup
| Item | Value |
|---|---|
| Data | CIFAR-10 training set (50,000 images), per-channel normalisation, no augmentation |
| Model | 3 × [Conv 3×3 → GeLU → AvgPool 2] (32, 64, 64 channels) → Linear(1024 → 10); 66,570 parameters; no BatchNorm or dropout. Smooth (C^∞) network, as Assumption 1 requires |
| Loss | Cross-entropy; reported as the full 50,000-image training loss every 100 iterations |
| Initialisation | 0.1 × PyTorch default init, same draw for both optimizers |
| Optimizers | SGD: θ ← θ − η∇L_n. Eq. (1): θ ← θ − η(∇L_n + Υ_n), Υ_n^i = ±ρ₁/(ρ₂ + (∂L_n/∂θ_i)²) |
| Hyper-parameters | η = 0.01 (constant), ρ₁ = 10⁻⁷, ρ₂ = 10⁻⁶, batch 128, 50 epochs = 19,500 iterations, no momentum or weight decay |
| Seeds | 0, 1, 2. A seed fixes θ₀ and the batch order, shared by both optimizers; the noise signs come from a separate generator |
| Landscape (panels 3–4) | Seed 0. L(θ* + αd₁ + βd₂) on a 31×31 grid, α, β ∈ [−1, 1], filter-normalised random directions (Li et al. 2018), 5,000 fixed training images, clipped at 10 for display |

## Results
| Seed | SGD final loss | SGD largest \|L − ln 10\| | Eq. (1) escape iteration (first L < ln 10 − 0.1) | Eq. (1) final loss |
|---|---|---|---|---|
| 0 | 2.30258 | 5.4·10⁻⁶ | 1,500 | 0.9586 |
| 1 | 2.30258 | 7.6·10⁻⁶ | 1,500 | 0.9508 |
| 2 | 2.30258 | 7.4·10⁻⁶ | 1,500 | 0.9444 |
| **Mean** | **2.30258** | | **1,500** | **0.9513** |

- **SGD:** the loss stays at ln 10 = 2.3026 (a uniform guess over 10 classes) for all 19,500 iterations; the full-set loss changes by less than 10⁻⁵.
- **Eq. (1):** leaves the plateau at iteration 1,500 on every seed; seed-to-seed spread is barely visible.
- **Landscapes (seed 0):**
  - SGD's final point sits on a flat surface at 2.303, between 2.3026 and 2.3026 over the whole grid.
  - Eq. (1)'s final point sits at the bottom of a bowl, with L(θ*) = 0.968 and a grid maximum of 18.3.
  - The perturbation directions are scaled to each run's own weights. SGD's weights are still tiny, so the two panels are on different scales and illustrate rather than compare.

## Interpretation and limits
- **Shown:** noise-assisted escape from a flat stationary region in a real network, reproducible across seeds, where SGD stalls within the budget.
- **Not shown, and not claimed:**
  - **That the starting region is a degenerate saddle of a specific order p.** θ = 0 is stationary (all hidden signals vanish), but p is not measured.
  - **That the final point lies in a PL or KL region.** The 2D landscape is a random slice and cannot establish PL membership. Final Experiment 2 tests that in a model where it can be certified.
  - **That the FIM scaling specifically is responsible.** At this initialisation 99.96% of coordinates have g² ≪ ρ₂, so Eq. (1)'s noise is effectively uniform here, as the paper itself notes on p.10.
  - **That SGD never escapes.** Only that it does not within 19,500 iterations.
- **Untested alternative explanation:** the noise may simply grow the small weights back to ordinary size. The weight norm was not logged.
- **Hyper-parameter selection:** ρ₁ was chosen in pilot runs on a max-pool version of this network (noise ratio 0.1 escaped within 4,000 iterations, 0.01 did not). SGD's learning rate was not tuned for this initialisation.
