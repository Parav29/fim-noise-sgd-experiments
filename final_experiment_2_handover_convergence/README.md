# Final Experiment 2 — Structural saddle-to-PL/KL handover and convergence

Figures: `fig_exp2_q2_PL.png` (PL destination), `fig_exp2_q4_KL.png` (KL destination). Numbers: `summary.json` (theory constants and per-configuration outcomes), `convergence_numbers.json` (rates and floors). Raw data: `raw/*.npz` (per-trajectory event times for all 200 trajectories, quantile curves, and 20 full trajectories per configuration).
Reproduce: `python3 run_experiment2.py && python3 make_figures_exp2.py` (about 1 minute on a laptop CPU, deterministic seeds).

## Question and short answers
The paper's central claim (abstract; Thms 3–6; Cor. 1) is the chain **degenerate saddle → noise-assisted escape → direct entry into a PL/KL region → retention → convergence**. All numbers are over 200 trajectories per configuration. PL means q = 2 and KL means q = 4.

| Question | Answer |
|---|---|
| Does noise enable escape from the degenerate saddle? | **Yes.** Eq. (1) escapes in 100% of trajectories, from the saddle and from the near-saddle start, with mean escape time 25–38 iterations. That is 0.09–0.13× the Thm 3 bound (331 PL / 269 KL). SGD never escapes (0%). At the exact saddle this is trivial (zero gradient, zero mini-batch noise). From (0.01, 0.01), plain gradient descent would need 624,664 (PL) / 312,241 (KL) iterations, versus the 50,000 run. |
| Does crossing the theoretical radius give direct entry into a certified PL/KL region? | **Yes, in 100% of trajectories** (Δτ = τ_handover − τ_escape ∈ [−229, 0]: entry at or before escape). **But this is guaranteed by the geometry of this model, not a stochastic test** (see "What the handover result does and does not show"). |
| Does the optimizer remain inside the region? | **Eventually, not immediately.** 71–78% of Eq. (1) trajectories leave the certified region at least once after first entering it, and **21–30% slide back through the saddle into the other basin.** Every such event happens within the first 430 iterations. After that, 100% stay inside for the remaining ~49,500 iterations. Variance-matched isotropic noise never settles with a constant step (about 1,000 exits per trajectory, still exiting at the end of the run). |
| Does convergence match the region's mathematics? | **Yes, when the noise is switched off after handover** (a diagnostic, not the paper's algorithm). PL: geometric, ln-slope −0.004004 per iteration, equal to the local rate 2 ln(1 − 0.2η) set by the smallest Hessian eigenvalue, and far faster than the certified PL rate ln(1 − ημ) = −8.7·10⁻⁵. KL: power law with log-log slope −2.08, matching the KL(κ = 3/4) prediction t^(−1/(2κ−1)) = t⁻². |
| Does persistent noise prevent exact convergence? | **Yes.** With a constant step, Eq. (1) settles at a loss floor of 0.028 (PL) / 0.027 (KL) that does not move between iterations 5k and 49k (0.0280 → 0.0281 → 0.0279). About 75–80% of the floor is in the flat b-direction. Isotropic noise floors are 8× / 6× higher (0.225 / 0.172). With the Robbins–Monro step the floor keeps falling (0.0034 / 0.0028 at the end). |

## Setup

### Model, data, loss
- **Network:** a tied 4-layer deep linear network, f_{a,b}(x) = (a⁴x, bx) (identity activations; the first output is four layers each with weight a). Targets y = (x, 0).
- **Data:** 4,096 inputs x ~ Uniform{−1, +1}.
- **Loss:** per-sample ℓ = 0.1[(f₁ − y₁)^q + (f₂ − y₂)²], so L_q(a, b) = 0.1[(a⁴ − 1)^q + b²], with q = 2 (PL destination) and q = 4 (KL destination, κ = 3/4). L* = 0 exactly.
- **Network check:** `torch_check` builds the actual network in PyTorch and confirms that autograd mini-batch gradients equal the analytic gradient on every batch.

### Optimizers (all with identical initialisation; 200 independent trajectories per configuration)
| Name | Update |
|---|---|
| SGD | θ ← θ − α_n ∇L_n (here identical to gradient descent; see assumption 1) |
| Eq. (1) | θ ← θ − α_n(∇L_n + Υ_n), Υ_n^i = ±ρ₁/(ρ₂ + (∂L_n/∂θ_i)²) w.p. ½ |
| Isotropic (control) | θ ← θ − α_n(∇L_n + ξ_n), ξ_n ~ N(0, σ²I), σ = ρ₁/ρ₂, i.e. the same variance as Eq. (1)'s noise at a zero gradient |
| Eq. (1), noise off after handover (diagnostic) | Eq. (1) until first entry into the certified region, then plain GD. **Not the paper's algorithm**; used only to measure the region's rate. |

- **Step rules:** constant α_n = η = 0.01, and Robbins–Monro α_n = η·n₀/(n₀ + n) with n₀ = 1,000 (Σα = ∞, Σα² < ∞, as in Thm 6).
- **Initialisations:** the exact saddle (0, 0), and the near-saddle point (0.01, 0.01).
- **Run length:** 50,000 iterations.

### Analytic verification before training (`theory()` in the code; all asserted)
| Property | q = 2 | q = 4 |
|---|---|---|
| Stationary points (sympy) | (0,0), (±1,0) | (0,0), (±1,0) |
| Derivatives along u = e_a at 0, orders 1–4 | 0, 0, 0, **−4.8** ⇒ p = 4 | 0, 0, 0, **−9.6** ⇒ p = 4 |
| Mixed 4th derivatives ∂⁴L/∂a^k∂b^(4−k) at 0 | all 0 (so L_p = 0) | all 0 |
| Hessian at saddle / at minimum | diag(0, 0.2) / diag(3.2, 0.2) | diag(0, 0.2) / diag(0, 0.2) (singular ⇒ not PL, KL only) |
| Linearisation radius r_lin (remainder ≤ ½ leading term) | 0.841 | 0.674 |
| K′_p, K_p (Eq. 31 with L_p = C₀ = 0), K_{p+1} | 0.476, 1.200, 0.252 | 1.187, 2.400, 0.400 |
| **Design radius r = r_lin/2** | **0.420** | **0.337** |
| Thm 4 constants a, b at r (R(r₀) = 1) | 0.360, 0.0141 | 1.440, 0.0103 |
| Destination constant μ from Thm 4 / Cor. 1 at r | **0.00871** (R_PL convention) | **0.240** (R_KL, κ = 3/4) |
| Certified region D | \|a\| ≥ a_c = 0.374 (any b) | \|a\| ≥ a_c = 0.317, \|b\| ≤ 15.5 |
| Exact radius along b = 0 | 0.374 | 0.299 |
| Analytic inf of ratio on D / numerical grid inf | 0.00871 / 0.00871 | 0.240 / 0.284 |
| ρ₃ = max g_a² on \|a\| ≤ r; ρ₂ = ρ₃; ρ₁ from Eq. (31) at r₀ = r | 0.00332, 0.00332, 0.0257 | 0.00347, 0.00347, 0.0244 |
| Eq. (35)/(41): η(ρ₁/(ρ₂+ρ₃))² vs threshold | 0.150 = 0.150 (at threshold) | 0.124 = 0.124 |
| Smoothness ℓ on \|a\| ≤ 1.2; η ≤ 1/ℓ | 13.3; 0.01 ≤ 0.075 | 74.6; 0.01 ≤ 0.0134 |
| Thm 3 escape bound E[τ] (c′ = ½, N = 0) | 331 iterations | 269 iterations |
| Exact GD escape time from (0.01, 0.01) | 624,664 | 312,241 |

**Certified regions (derivation used in `ratio_lower_bound`).** Write A = (a⁴ − 1)^q and B = b².
- **PL (q = 2):** R_PL = ‖∇L‖²/(2L) = (3.2a⁶A + 0.2B)/(A + B). This is a weighted average of 3.2a⁶ and 0.2, so R_PL ≥ min(3.2a⁶, 0.2), with equality attainable at b = 0. Hence on {|a| ≥ a_c} the infimum is exactly μ = 3.2a_c⁶.
- **KL (q = 4):** R_KL² = (2.56a⁶A^1.5 + 0.04B)/(0.1^1.5 (A + B)^1.5). Using (A + B)^1.5 ≤ √2(A^1.5 + B^1.5) and 0.04B ≥ (0.04/|b|)B^1.5 gives R_KL ≥ √(min(2.56a⁶, 0.04/|b|)/(0.1^1.5√2)). b_c = 15.5 is the largest |b| for which the a-term is still the binding one.
- Both bounds were also checked against the exact ratio on a dense grid inside D. Membership in D (not the pointwise ratio) is what defines handover; Panel D shows why. Away from b = 0 the measured ratio exceeds μ before certified entry.

## Results

### Outcome table (percent of 200 trajectories; Eq. (1) and isotropic rows; SGD: 0% escape in every configuration)
| Config | Escaped | Immediate handover | Eventual / failed | ≥1 region exit | Saddle re-entry | Basin switch | No exit after n = 5,000 | Loss floor (median, last 10k it.) |
|---|---|---|---|---|---|---|---|---|
| PL, saddle, Eq. (1) const | 100 | 100 | 0 / 0 | 78.0 | 76.0 | 24.5 | 100 | 0.0284 |
| PL, saddle, Eq. (1) RM | 100 | 100 | 0 / 0 | 73.0 | 75.0 | 23.5 | 100 | 0.0034 |
| PL, saddle, iso const | 100 | 100 | 0 / 0 | 100 | 100 | 100 | 0 | 0.225 |
| PL, saddle, iso RM | 100 | 100 | 0 / 0 | 100 | 100 | 98.0 | 65.5 | 0.0061 |
| PL, saddle, Eq. (1) noise-off diag. | 100 | 100 | 0 / 0 | 0 | 0 | 0 | 100 | 0 (to float precision) |
| PL, near, Eq. (1) const | 100 | 100 | 0 / 0 | 77.5 | 79.0 | 29.5 | 100 | 0.0286 |
| KL, saddle, Eq. (1) const | 100 | 100 | 0 / 0 | 71.0 | 68.5 | 22.5 | 100 | 0.0276 |
| KL, saddle, Eq. (1) RM | 100 | 100 | 0 / 0 | 72.0 | 69.5 | 22.0 | 100 | 0.0028 |
| KL, saddle, iso const | 100 | 100 | 0 / 0 | 100 | 100 | 100 | 0 | 0.172 |
| KL, saddle, iso RM | 100 | 100 | 0 / 0 | 99.5 | 99.0 | 93.0 | 76.0 | 0.0036 |
| KL, near, Eq. (1) const | 100 | 100 | 0 / 0 | 71.5 | 71.0 | 21.0 | 100 | 0.0277 |

The remaining near-saddle rows are in `summary.json` and match the saddle rows to within a few percent. No trajectory diverged in any configuration.

Definitions:
- **Escape:** first n with |Z_n| = |⟨θ_n − θ_deg, u⟩| = |a_n| ≥ r, which is the paper's τ.
- **Immediate handover:** θ at the escape iteration is in D.
- **Region exit:** a transition from inside D to outside, after first entry.
- **Saddle re-entry:** a return to |a| < r after escape.
- **Basin switch:** a sign change of a after handover, i.e. the trajectory crossed the saddle line a = 0 into the other minimum's basin.

### What the handover result does and does not show
- The paper's sufficient radius r (0.420 / 0.337) lies **outside** the certified boundary a_c (0.374 / 0.317). So the paper's Thm 4 / Cor. 1 radius is **valid and conservative** here, by 12% (PL) and 6% (KL, against the certified bound; 13% against the exact b = 0 radius).
- Because this model's PL region depends only on a (and the KL box |b| ≤ 15.5 is never approached), every trajectory that reaches |a| = r is already in D. **The 100% immediate-handover figure is therefore a consequence of the geometry. It confirms the computation of the radius, but it could not have come out otherwise in this model.** The non-trivial measured outcomes are escape (and its time), retention, the convergence rate, and the noise floor.

### Retention (Thm 5, Thm 6)
- **Slide-backs do occur.** With the theory-chosen noise (exactly at the Eq. (35)/(41) threshold), the noise kick per step at the region boundary is 0.052 (PL) / 0.041 (KL). That is 126× / 84× the gradient drift there (0.00041 / 0.00049). Trajectories therefore wander back across a_c, and about a quarter cross the saddle into the other basin.
- **All of this happens in a short transient:** the last region exit is at most iteration 339 (PL, constant), 430 (PL, RM), 327 (KL, constant), or 352 (KL, RM). After that, every Eq. (1) trajectory stays in D until iteration 50,000, and all end inside D. The highest end-of-run loss is 0.034, while any point outside D has loss ≥ 0.096. The 5th percentile of the final |a| is 0.79–0.97, against a_c = 0.32–0.37.
- **The FIM scaling is what makes retention happen.** The isotropic control with the same noise at zero gradient escapes equally fast, but under a constant step it never stays in D. Under RM it settles in only 62–78% of trajectories by n = 5,000, with exits up to iteration 27,075. Eq. (1)'s noise shrinks wherever the gradient is large (inside the basin in the a-direction); isotropic noise does not.

### Convergence (Panels E, E′)
- **Noise off after handover:**
  - PL: geometric decay with slope −0.004004 per iteration (k = 2,000–6,000). This equals 2 ln(1 − 0.2η), the rate of the b-direction (smallest Hessian eigenvalue 0.2), and is much faster than the certified worst case (1 − ημ)^k.
  - KL: power-law decay, log-log slope −2.08 (k = 15,000–45,000), matching t⁻².
  - (The PL curve reaches about 10⁻³⁰ by k ≈ 15,000, so later PL slopes are floating-point artifacts and are not used.)
- **Eq. (1) with a constant step (the paper's analysed algorithm):**
  - It does not converge to L*. The loss settles at a nonzero noise floor of about 0.028, which is flat in time.
  - The floor sits mostly in the flat b-direction: the median final |b| is 0.43–0.47, and on the 20 stored trajectories the b-term 0.1b² is 74–81% of the final loss. There ∂L/∂b = 0.2b is small, so Eq. (1)'s noise stays near its maximum ρ₁/ρ₂.
- **Eq. (1) with the Robbins–Monro step:** the floor keeps decreasing (log-log slope vs accumulated step −1.14 PL / −1.19 KL near the end). This is consistent with convergence as α_n → 0, but no rate claim is made.

## Assumptions and design choices (all fixed before the runs)
1. **There is no mini-batch noise (M_{n+1} ≡ 0).** Because x² = x⁴ = 1, every sample has the same loss, so SGD is exactly gradient descent and the 4,096-point dataset and the batch size do not matter. Claim 1 holds trivially. SGD started *at* the saddle cannot move; the near-saddle start is the meaningful SGD comparison.
2. **Paper constants that the draft leaves open were fixed by stated rules:**
   - r_lin is the largest radius where the Taylor remainder is at most half the leading term;
   - r = r_lin/2;
   - c′ = ½;
   - ρ₂ = ρ₃ (noise halves where g_a² = ρ₃);
   - ρ₁ is set so that Eq. (31)'s radius equals r exactly (the Eq. (35)/(41) threshold);
   - η = 0.01, which satisfies η ≤ 1/ℓ on |a| ≤ 1.2;
   - n₀ = 1,000 ≥ the Thm 3 bound.

   The outcome values were not tuned. The only rerun added a |b| statistic, with identical seeds and identical results.
3. **PL convention.** Thm 2 writes ‖∇L‖² ≥ 2μ(L − L*) but Thm 4 writes ‖∇L‖² ≥ μ(L − L*). We use the factor-2 form (R_PL) and convert μ_Thm4 = 2μ.
4. **Cone condition.** Lemma 2 / Thm 4 use the cone condition ‖v‖ ≤ C₀|Z|^(p−1). The noisy iterates do not satisfy it: b is driven by noise, not slaved to a. It is not needed here because all mixed p-th derivatives vanish (L_p = 0), so we set C₀ = 0. The paper's loss expansion in Thm 4 omits the Hessian term 0.1b². We evaluate its constants along b = 0, which is what the cone condition implies. The certified region does not rely on this.
5. **ρ₃** is taken as a bound on (∂L/∂θ_i)² along u, which is what the lower bound ρ₁/(ρ₂ + ρ₃) requires. The draft's sentence below Eq. (11) describes it as a bound on Υ.
6. **Isotropic control.** Gaussian noise with σ = ρ₁/ρ₂, matched to Eq. (1) at zero gradient. Its variance also satisfies the theorems' escape condition E⟨ξ, u⟩² ≥ (ρ₁/(ρ₂ + ρ₃))².
7. **Thm 5 and Thm 6 are statements about limits and probabilities** (η → 0, exponential residence time; 1 − O(Σα²)). We report observed frequencies over 200 trajectories and 50,000 iterations, not those limits.

## Agreement with the paper
| Paper statement | Result |
|---|---|
| Polynomial-time escape, E[τ] ≤ Thm 3 bound | **Consistent.** Measured mean ≈ 0.1 × bound |
| Handover radius (Thm 4 / Cor. 1) is sufficient for PL/KL entry | **Consistent**, and conservative by 6–13% in this model |
| Entering PL needs more noise than entering KL | **Not tested.** μ is not comparable across q here, so the two thresholds cannot be ranked |
| No slide-back after handover (abstract, Sec. 5) | **Only after a transient.** 21–30% of trajectories slide back through the saddle within the first 430 iterations at η = 0.01. Thm 5 is an η → 0 statement, so this does not contradict the theorem, but the abstract's "remain locked … avoiding slidebacks" should not be read as immediate |
| Linear rate in a PL region, sublinear in a KL region (p.4, p.14) | **Consistent** when the noise is off after handover (geometric vs t⁻²). The paper's constant-step algorithm itself stops at a noise floor |
| Eq. (1) "converges to a local minimum" (p.4) | **Only with a decaying step (Thm 6).** With a constant step there is a persistent floor |
| FIM-scaled (anisotropic) noise | Escape: no advantage over isotropic noise. **Retention and floor: a large advantage** |

## Limits of this experiment
- The model is two-dimensional and separable, so the handover geometry is simple and the 100% immediate-handover result is structural.
- There is a single design point (r = r_lin/2, noise at the threshold). Below-threshold noise levels were not run.
- The near-saddle start is one fixed point, not a distribution.
