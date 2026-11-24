# SGD vs. FIM-scaled-noise SGD (Eq. 1) on CIFAR-10

There are two experiments. **Experiment 1** uses the standard init, where the noise has no visible effect. **Experiment 2** starts near the degenerate saddle at θ = 0, where plain SGD stalls and Eq. (1) escapes.

Files: `fim_noise_sgd.py` (both optimizers + training), `plot.py` (figure), `results.json` (raw curves + landscape grids), `train.log`, `results_convergence_landscape.png`.
Reproduce: `python3 fim_noise_sgd.py && python3 plot.py`

## Experiment 1: standard initialization

### Setup
| | |
|---|---|
| Data | CIFAR-10 train set (50,000 images), per-channel normalization, no augmentation |
| Model | 3×[Conv3×3 → GeLU → MaxPool2] (32, 64, 64 channels) → Linear(1024→10); 66,570 params; no BN, no dropout |
| Loss | cross-entropy |
| Exp. A | θ ← θ − η ∇L_n (`torch.optim.SGD`, no momentum, no weight decay) |
| Exp. B | θ ← θ − η (∇L_n + Υ_n), Υ_n^i = s_i ρ₁ / (ρ₂ + (∂L_n/∂θ_i)²), s_i = ±1 w.p. ½ (`FIMNoiseSGD`) |
| η | 0.01, constant (α_n = η) |
| Batch / length | 128, 50 epochs = 19,500 iterations (last partial batch dropped) |
| ρ₁, ρ₂ | 1e-9, 1e-6 |
| Seeds | seed 0. Identical θ₀ and identical mini-batch sequence for both runs; the signs s_i come from a separate generator |
| Hardware | Apple M4 (MPS). The same script also ran on a Kaggle T4 (CUDA) and gave final losses of 0.5045 (A) and 0.5052 (B), the same picture |

**How ρ₁ and ρ₂ were chosen** (one measurement, no sweep): on one mini-batch at θ₀, |∂L/∂θ_i| has median 2.5e-4, 90th percentile 8.8e-4, and 99th percentile 2.4e-3.
- ρ₂ = 1e-6 ≈ g² at the 90th–95th percentile. Coordinates with large gradients get damped noise, and coordinates with small gradients (flat directions) get noise near the maximum.
- ρ₁/ρ₂ = 1e-3 is the maximum noise per coordinate, about 4× the median |g|. At θ₀ the noise vector's norm (≈0.26) is about twice the gradient's norm.

**Loss recorded:** the mini-batch loss at every iteration (thin lines), and the full 50k training-set loss every 100 iterations (bold lines; this is the curve used for plot 2).

**L_ref = 0**, the infimum of cross-entropy. I fixed it before seeing any results. I did not use "lowest loss observed" because that makes the log gap plunge to −∞ at the end of training and looks like a fake linear-rate phase.

### Results (`results_convergence_landscape.png`, top row)
- **The two runs are practically identical.** The largest gap between their full training-set losses is 0.004. Final loss is 0.5042 for SGD and 0.5057 for FIM-noise. Mean mini-batch loss over the last epoch is 0.5318 vs 0.5317.
- **There is no long slow phase for the noise to escape.** Plain SGD sits near ln 10 ≈ 2.30 for only about 200 iterations, then descends steadily. Both methods leave that plateau at the same time.
- **Log gap:** the slope of log L per 1k iterations is −0.18 (0–300), −0.26 (300–2k), −0.11 (2k–5k), −0.056 (5k–10k), −0.041 (10k–15k), and −0.045 (15k–19.5k), with identical values for both methods.
  - The rate decays and then roughly levels off at a small constant slope over the last ~10k iterations.
  - That is consistent with sublinear progress followed by a slow, roughly linear-in-log tail. However, the tail is slow, is measured against an assumed L_ref, and appears equally in plain SGD. So it cannot be credited to the noise, and it is not clear evidence of a saddle-to-PL handover.

### Loss landscapes (`results_convergence_landscape.png`, bottom row)
The method is the same as Method A in `../loss_landscape_pl_region/landscape.py`:
- **Directions:** L(θ* + α d₁ + β d₂) around each run's final weights θ*. d₁ and d₂ are random Gaussian directions, filter-normalised (Li et al. 2018), with a fixed seed. The raw draws are the same for both runs, each normalised to its own θ*. Biases get direction 0.
- **Grid:** α, β ∈ [−1, 1] on a 31×31 grid.
- **Loss:** evaluated on a fixed subset of 5,000 training images and clipped at 10 for plotting.
- **Weights:** the final weights come from re-running the same script. The re-run reproduced the earlier loss curves exactly (difference 0.0 at every checkpoint).

What the landscapes show:
- **Both runs ended in a single bowl-shaped basin,** with the minimum of the grid at θ* itself. The loss there is 0.516 (SGD) and 0.518 (FIM-noise).
- **The two landscapes are practically the same.** The highest loss within |α|, |β| ≤ 0.25 is 0.93 vs 0.94; within ≤ 0.5 it is 2.86 vs 2.87; at the edges it is about 16 for both. The largest pointwise difference across the whole grid is 0.13, at the steep edges.
- **So the noise did not lead to a flatter or different basin at this setting.** A 2D random slice can't show saddles or flat directions outside the plane, so this is a picture of the local geometry, not a test of the PL condition.

### Limitations
- **Why the noise does nothing at this setting:**
  - Each step moves a coordinate by at most η·ρ₁/ρ₂ = 1e-5, with a random, zero-mean sign. Over N steps that grows like √N, while the gradient drift grows like N.
  - The mini-batch gradient noise SGD already has is of similar or larger size.
  - The paper's guarantees concern behaviour near strict or degenerate saddles. This trajectory never visibly stalls at one, so the mechanism has nothing to act on.
- **Not tested, per the scope restriction:** a larger noise-to-step-size ratio (the paper says entering a PL region needs a larger ρ₁/(ρ₂η) ratio), other learning rates, initializations that start in a flatter region, and multiple seeds. With a single seed, differences of order 0.004 are within run-to-run noise.

## Experiment 2: start near the saddle (small initialization)
Files: `results_smallinit.json` (seed 0, with landscapes), `results_smallinit_seed{1,2}.json`, `train_smallinit.log`, `results_smallinit_convergence_landscape.png`.
Reproduce: `python3 fim_noise_sgd.py --init_scale 0.1 --rho1 1e-7 --out results_smallinit.json && python3 plot.py results_smallinit.json`

### Why this setting
- **θ = 0 is a degenerate saddle.** At θ = 0 every hidden activation and every back-propagated signal is zero, so θ = 0 is a stationary point of this 4-weight-layer network. Small-init training starts next to it (Saxe et al. 2014; Jacot et al. 2021; Bantzis et al. 2026).
- **SGD's own noise vanishes there too.** The gradient of every individual sample goes to zero at this saddle. Ziyin et al. (2023) call this a Type-II saddle and show SGD can stall at such saddles indefinitely.
- **This is Eq. (1)'s target case.** Υ = ρ₁/(ρ₂ + g²) is largest exactly where g → 0. That matches the paper's motivation that gradient information alone cannot escape degenerate saddles (Anandkumar & Ge).

### What changed from Experiment 1 (nothing else did)
| | Exp. 1 | Exp. 2 |
|---|---|---|
| θ₀ | default PyTorch init | 0.1 × default init (same draw, scaled) |
| ρ₁ (ρ₂ = 1e-6 in both) | 1e-9 (maximum noise ρ₁/ρ₂ = 1e-3) | 1e-7 (maximum noise ρ₁/ρ₂ = 0.1, so ηρ₁/ρ₂ = 1e-3 per step) |

### How these values were chosen
I ran two short pilots (4,000 iterations, seed 0) and report all of them. This was a targeted search, not a blind sweep.
- **Pilot 1, init scale κ ∈ {1, 0.3, 0.1, 0.03} with the old ρ:**
  - At κ = 1 both methods escape at about iteration 230.
  - At κ ≤ 0.3 both are stuck at ln 10 for all 4,000 iterations.
- **Pilot 2, κ ∈ {0.3, 0.1} × (ρ₁, ρ₂) ∈ {(1e-8, 1e-6), (1e-7, 1e-6), (1e-10, 1e-8), (1e-9, 1e-8)}:**
  - Only the two settings with maximum noise ρ₁/ρ₂ = 0.1 escaped, at iterations 1,100–1,650.
  - The ρ₁/ρ₂ = 0.01 settings stayed stuck.
  - This matches the paper's statement that escape needs a large enough noise-to-step-size ratio.

### Results (3 seeds; seed 0 on MPS, seeds 1–2 on a Kaggle T4)
| seed | SGD final loss | SGD max deviation from ln 10 over 19,500 iterations | FIM-noise: first check with L < 2.2 | FIM-noise final loss |
|---|---|---|---|---|
| 0 | 2.3026 | 5.4e-6 | iteration 1,400 | 0.593 |
| 1 | 2.3026 | 7.6e-6 | iteration 1,200 | 0.583 |
| 2 | 2.3026 | 7.4e-6 | iteration 1,400 | 0.586 |

- **Escape:** plain SGD never leaves the saddle plateau in 50 epochs on any seed. Eq. (1) escapes after about 1,200–1,400 iterations on every seed, then converges to a loss of about 0.59.
- **Log gap (plot 2, seed 0):** after escape, the slope of log L per 1k iterations is −0.25 (1k–2.5k), −0.11 (2.5k–5k), −0.067 (5k–10k), −0.043 (10k–15k), and −0.039 (15k–19.5k). It settles to a roughly constant rate, the same shape as Experiment 1's normal-init run, delayed by the escape time.
  - A roughly linear-in-log tail after escape is consistent with the paper's handover picture.
  - With an assumed L_ref = 0 and one 2D view, it is not proof of a PL region.
- **Landscapes (seed 0):**
  - SGD's θ* is still at the saddle, and the plane through it is flat at 2.3026 everywhere. The filter-normalised directions are scaled to SGD's still-tiny weights, so this plot shows only a small neighbourhood of the saddle.
  - FIM-noise's θ* sits at the bottom of a bowl-shaped basin, with L(θ*) = 0.609 on the 5k subset.

### Limitations of Experiment 2
- **The stall comes from the initialization.** It was created by shrinking the init, which is a standard way to start near a saddle but not the default practice.
- **Larger noise costs some final loss.** After escape, FIM-noise ends at about 0.59 vs 0.504 for SGD from the normal init in Experiment 1, so the stronger noise slows the in-basin phase somewhat.
- **Plain SGD may escape eventually.** The run budget is 19,500 iterations; I did not test how long plain SGD takes to escape, or whether it ever does.

### References
- Saxe, McClelland, Ganguli (2014), arXiv:1312.6120
- Jacot et al. (2021), arXiv:2106.15933
- Bantzis, Simon, Jacot (ICLR 2026), arXiv:2505.21722
- Ziyin, Li, Galanti, Ueda, "Type-II Saddles and Probabilistic Stability of SGD", arXiv:2303.13093
- Du et al. (2017), arXiv:1705.10412
