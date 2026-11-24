# Final conclusion: experimental check of FIM-scaled noise SGD (Eq. 1)

**Paper:** Ramaswamy & Borkar, "Polynomial-Time Handovers from Strict and Degenerate Saddles to Local Polyak–Łojasiewicz Basins in Deep Learning".

**Short answer:**
- **The paper's main claim was confirmed in the setting it is about.** When training starts next to a degenerate saddle, plain SGD stays stuck for all 50 epochs. The FIM-noise method of Eq. (1) escapes after about 1,200–1,400 iterations on all 3 seeds, settles into a basin, and keeps converging there.
- **Away from saddles, the noise does nothing.** With a standard initialization the two methods give identical curves, because nothing slows plain SGD down.

---

## 1. What we are using, and how

### Data, model, loss
| Item | Value |
|---|---|
| Dataset | CIFAR-10 training set (50,000 images) from `torchvision.datasets.CIFAR10`, normalised per channel, no augmentation |
| Model | 3 × [Conv 3×3 → GeLU → MaxPool 2] with 32, 64, 64 channels → Linear(1024 → 10); 66,570 parameters; no BatchNorm, no Dropout |
| Loss | Cross-entropy |
| Training | Learning rate η = 0.01 (constant), batch size 128, 50 epochs = 19,500 iterations, no momentum, no weight decay |

### The two optimizers
- **Exp. A, standard SGD:** θ ← θ − η ∇L_n, using `torch.optim.SGD`.
- **Exp. B, FIM-noise SGD (Eq. 1 of the paper):** θ ← θ − η (∇L_n + Υ_n), where Υ_n^i = s_i · ρ₁ / (ρ₂ + (∂L_n/∂θ_i)²).
  - s_i = ±1 with probability ½, drawn independently for every parameter at every step.
  - It is implemented as the `FIMNoiseSGD` class in `fim_noise_sgd.py`. A built-in self-check verifies the update matches the formula.

### Fairness controls
- Both methods use the same initial weights θ₀ and the same sequence of mini-batches.
- The random signs come from a separate random generator, so they don't change the mini-batch order.
- All seeds are fixed in advance.
- Re-running the code reproduces the earlier curves exactly (difference 0.0 at every checkpoint).

### What we measure
- **Training loss:**
  - the mini-batch loss at every iteration;
  - the full 50,000-image training-set loss every 100 iterations.
- **Log loss gap:** log(L(θ_n) − L_ref) with L_ref = 0, as discussed in Section 4.
- **Loss landscape:**
  - **Plane:** L(θ* + α d₁ + β d₂) for α, β ∈ [−1, 1] on a 31×31 grid, around each run's final weights θ*.
  - **Directions:** d₁ and d₂ are random Gaussian directions with a fixed seed, filter-normalised following Li et al. 2018. Biases are not perturbed.
  - **Evaluation:** the loss is computed on a fixed subset of 5,000 training images.
  - This is the same method as Method A in `../loss_landscape_pl_region/landscape.py`.

### Compute
- Apple M4 GPU (MPS) locally.
- A Kaggle T4 GPU for a cross-check of Experiment 1 and for seeds 1–2 of Experiment 2.

### Files
| File | Contents |
|---|---|
| `fim_noise_sgd.py` | Both optimizers, training, landscape |
| `plot.py` | The four-panel figure |
| `results_convergence_landscape.png` | Experiment 1 figure |
| `results_smallinit_convergence_landscape.png` | Experiment 2 figure |
| `results*.json`, `train*.log` | Raw data and logs |
| `RESULTS.md` | Detailed per-experiment notes |

---

## 2. Issues we faced, in order, and how we solved them

**Issue 1: with a standard setup, the noise made no difference (Experiment 1).**
- **Setup:** default PyTorch init, ρ₁ = 1e-9, ρ₂ = 1e-6. These values were calibrated from the measured gradient sizes at init: median |g| = 2.5e-4, 90th percentile 8.8e-4.
- **What happened:**
  - The two loss curves were the same to within 0.004.
  - Final loss was 0.5042 (SGD) vs 0.5057 (FIM-noise).
  - The two loss landscapes were also practically the same, both bowl-shaped around θ*.
- **Diagnosis:**
  - **No saddle:** plain SGD never got stuck. It left the initial plateau near ln 10 ≈ 2.30 after only about 200 iterations, so there was nothing for the noise to escape.
  - **Similar noise already present:** mini-batch SGD has its own gradient noise, which was about as large as the injected noise.
  - **Noise too weak to add up:** the injected noise moved each weight by at most ηρ₁/ρ₂ = 1e-5 per step, with a random sign. Over N steps that random walk grows only like √N, while the gradient's push grows like N.
- **Conclusion:** the model is non-convex, but non-convexity alone doesn't mean training will stall. This setting never put the optimizer in the situation the paper is about.

**Issue 2: a 32-hidden-layer network did not train at all (abandoned).**
- **Default init:** the signal died completely. Logits were identical for every input (spread about 1e-9), and the first layer's gradient was exactly 0.
- **Kaiming (He) init:** the signal still shrank about 10⁵× through the 32 layers. Both methods stayed at exactly ln 10 = 2.3026 for all 19,500 iterations.
- **Why the noise didn't help:**
  - The noise was still the weak setting (1e-5 per weight per step).
  - Escaping needs many layers to change together, which independent random kicks to individual weights can't produce.
- **Decision:** the user asked to drop this experiment, so it was stopped and its outputs deleted.
- **Lesson:** to test the paper fairly we need a saddle that is controlled and well understood, plus noise strong enough to matter.

**Issue 3: making the run reproducible and running it.**
- **Slow download:** CIFAR-10 downloads from Toronto were very slow, so on Kaggle we used a hosted copy (`pankrzysiu/cifar10-python`).
- **GPU queue:** Kaggle's GPU queue was long, so most runs were done locally.
- **Low battery:** late in the work the Mac was on 7% battery and running about 10× slower. Seeds 1–2 of Experiment 2 were therefore run on Kaggle.

**Solution: Experiment 2, starting next to a degenerate saddle.**
- **Idea from the literature (Section 6):**
  - With small initial weights, training starts next to θ = 0. At θ = 0 every hidden activation and back-propagated signal is zero, so it is a stationary point, and for a network with 4 weight layers it is a degenerate saddle.
  - At this saddle the gradient of every individual image also goes to zero, so SGD's own noise vanishes. Ziyin et al. call this a Type-II saddle.
  - The paper's noise Υ = ρ₁/(ρ₂ + g²) is largest exactly when g → 0.
- **What changed (only these two things):**
  - θ₀ = 0.1 × (the same default init draw).
  - ρ₁ raised to 1e-7, so the maximum noise ρ₁/ρ₂ = 0.1 and ηρ₁/ρ₂ = 1e-3 per step. ρ₂ = 1e-6 is unchanged.
- **How the values were chosen:** two short pilots of 4,000 iterations each. Every configuration tried is reported in `RESULTS.md`.
  - Init scale ≤ 0.3 made both methods stall with the old noise.
  - Only a maximum noise of 0.1 escaped; 0.01 did not.

---

## 3. Results, and whether they match the paper

### Experiment 2 (3 seeds)
| seed | plain SGD | FIM-noise: first check with L < 2.2 | FIM-noise final loss |
|---|---|---|---|
| 0 | 2.3026 for all 19,500 iterations (max deviation 5.4e-6) | iteration 1,400 | 0.593 |
| 1 | 2.3026 throughout (max deviation 7.6e-6) | iteration 1,200 | 0.583 |
| 2 | 2.3026 throughout (max deviation 7.4e-6) | iteration 1,400 | 0.586 |

### Paper claims vs what we observed
| Paper's claim | What we observed | Match? |
|---|---|---|
| FIM-scaled noise escapes strict and degenerate saddles where plain gradient methods struggle (Anandkumar & Ge) | Plain SGD never left the saddle plateau in 50 epochs on any seed; Eq. (1) escaped in 1,200–1,400 iterations on every seed | **Yes**, qualitatively |
| Escape takes polynomial time | Escape was fast (about 1.3k iterations), but we did not vary the problem size, so the polynomial scaling itself was not tested | Not tested |
| Escape and PL entry need a large enough noise-to-step-size ratio ηρ₁/(ρ₂ + ρ₃) | Maximum noise 0.01 did not escape; 0.1 did, at both κ = 0.3 and κ = 0.1 | **Yes**, consistent |
| After the handover, convergence is linear (PL) or sublinear (KL) | Slope of log L per 1k iterations after escape: −0.25 → −0.11 → −0.067 → −0.043 → −0.039. It decays and then settles at a roughly constant small slope, consistent with sublinear progress turning into a slow, roughly linear tail | Consistent, but **not proof**; see limitations |
| No slide-back into the saddle after the handover | After escape the loss never returned to the plateau. The highest post-escape loss was about 2.06–2.11, right at the escape point, and the largest rise above the best loss so far was 0.13 / 0.07 / 0.06 (seeds 0/1/2) | **Yes** |
| The handover ends in a local minimum basin | FIM-noise's θ* is at the bottom of a bowl-shaped basin in the 2D landscape; SGD's θ* is still on a flat plateau at 2.3026 | **Yes**, within a 2D view |

### Experiment 1 vs the prompt's expectation
- **Expectation:** the noise would help escape slow regions and then converge faster.
- **Observed:** no difference, because there was no slow region (see Issue 1).
- **Is this a contradiction?** No. The paper's guarantees are about behaviour near saddles; they don't promise a speed-up when there's no saddle. In that case the noise averages to zero and is effectively neutral.

---

## 4. Assumptions
1. **L_ref = 0 in the log-gap plot.** This is the lowest value cross-entropy can take, fixed before seeing any results. The true local-minimum loss is unknown and positive, so the true gap is smaller than plotted, and the "linear tail" is measured against an assumed reference.
2. **The small-init saddle stands in for the paper's saddles.** We assume the region around θ = 0 is a representative degenerate saddle. The paper is general and doesn't say which saddles a network will meet.
3. **Constant step size α_n = η.** The paper's constant-step-size analysis applies; we did not test diminishing step sizes.
4. **ρ₃.** The paper's ρ₃ (a bound on the noise near the saddle) is a quantity in the analysis, not something the algorithm sets, so it doesn't appear in the code.
5. **Seeds and noise.** We treat 3 seeds as enough to show the escape isn't a one-off, and differences below about 0.01 in loss as run-to-run noise.
6. **Full training-set loss is the right curve to analyse.** The mini-batch loss is too noisy to read rates from.

---

## 5. Limitations
- **The saddle was created by our choice of initialization.** Shrinking the init to 0.1× is a standard way to start near a saddle, but it isn't default practice. With the default init (Experiment 1) the method gives no benefit.
- **The noise level was tuned.** The successful ρ₁ came from a small targeted pilot (8 configurations), not from theory. A 10× smaller noise didn't escape at all.
- **Larger noise costs some final loss.** After escape, FIM-noise ends at about 0.59, vs 0.504 for SGD from the normal init (Exp. 1). Strong noise keeps acting in flat directions inside the basin, which slows the final phase.
- **Short budget for plain SGD.** We didn't run plain SGD beyond 19,500 iterations, so we can't say whether it escapes eventually or never. We can only say it doesn't within 50 epochs.
- **The landscape is only a 2D slice.** It can't show saddles or flat directions outside the plane, and it isn't a test of the PL condition. SGD's landscape in Exp. 2 is flat partly because the directions are scaled to SGD's still-tiny weights.
- **The rate claims weren't tested formally.** We didn't check the PL inequality ‖∇L‖² ≥ 2μ(L − L*), estimate the degeneracy order p, or measure the handoff radius r.
- **One model, one dataset.** A small 3-layer CNN on CIFAR-10, a learning rate of 0.01, and 3 seeds for Exp. 2 (1 seed for Exp. 1, plus a CUDA cross-check). The 32-layer attempt was abandoned (Issue 2).

---

## 6. Sources
- **The paper:** Ramaswamy & Borkar, *Polynomial-Time Handovers from Strict and Degenerate Saddles to Local Polyak–Łojasiewicz Basins in Deep Learning* (`Handover_from_degenerate_to_PL_and_Oseledets.pdf`). Eq. (1) is in Section 2, and the noise-to-step-size discussion is in Sections 2–3.
- **Saddle at the origin under small initialization:**
  - Saxe, McClelland & Ganguli (2014). *Exact solutions to the nonlinear dynamics of learning in deep linear neural networks.* [arXiv:1312.6120](https://arxiv.org/abs/1312.6120)
  - Jacot, Ged, Şimşek, Hongler & Gabriel (2021). *Saddle-to-Saddle Dynamics in Deep Linear Networks: Small Initialization Training, Symmetry, and Sparsity.* [arXiv:2106.15933](https://arxiv.org/abs/2106.15933)
  - Bantzis, Simon & Jacot (ICLR 2026). *Saddle-To-Saddle Dynamics in Deep ReLU Networks: Low-Rank Bias in the First Saddle Escape.* [arXiv:2505.21722](https://arxiv.org/abs/2505.21722)
- **SGD's own noise vanishing at the saddle:** Ziyin, Li, Galanti & Ueda (2023). *Type-II Saddles and Probabilistic Stability of Stochastic Gradient Descent.* [arXiv:2303.13093](https://arxiv.org/abs/2303.13093)
- **Plain gradient descent being slow at saddles, and perturbations helping:** Du, Jin, Lee, Jordan, Singh & Póczos (NeurIPS 2017). *Gradient Descent Can Take Exponential Time to Escape Saddle Points.* [arXiv:1705.10412](https://arxiv.org/abs/1705.10412)
- **Landscape method:** Li, Xu, Taylor, Studer & Goldstein (NeurIPS 2018). *Visualizing the Loss Landscape of Neural Nets.* [arXiv:1712.09913](https://arxiv.org/abs/1712.09913)
- **Cited by the paper, not consulted directly here:**
  - Anandkumar & Ge (2016), on degenerate saddles.
  - Jin et al. (2017), perturbed gradient descent.
  - Vlaski et al., showing mini-batch noise is enough for strict saddles.
