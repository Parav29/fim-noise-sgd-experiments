# Experiment 2 — extended ablation: saddle degree, step size, noise level, escape radius

This is an extension of Final Experiment 2, not a new experiment. Same model family, same code (generalised), same seeds.

**Reproduce**, in this order: `python3 check_regression.py` (gate; about 1 min) → `python3 run_ablation.py` (about 6 min on 9 cores) → `python3 analyze_ablation.py` (about 10 s).
- **Outputs:** `results/` (one JSON per configuration, `table_all_configs.csv`, `theorem_checks.json`, `convergence_rates.json`), `raw/` (one `.npz` per simulation, 146 MB), `figures/` (PNG and PDF), `run_log.txt`. The pilot is in `pilot/` and `pilot_log.txt`.
- **Traceability:** every number below comes from a field in `results/<name>.json` → `summary` or `theory`. The per-trajectory arrays are in the `raw/` file named in that JSON's `config.raw_file`, with event prefix `config.event_prefix`.

## Professor-facing summary

- **Escape works and the bounds hold.**
  - Eq. (1) escaped in all 520 noisy configurations (200 trajectories each, none censored).
  - Every configuration that meets the theorem's assumptions escaped below the Thm 1 / Thm 3 bound: 78 at p = 2 and 162 at p ≥ 4. The largest measured/bound ratio is 0.55.
  - Plain SGD never left the exact saddle at any p. From (0.01, 0.01) it escapes in 910 / 403 iterations at p = 2 (strict saddle), but would need 3·10⁵ to 10¹³ iterations at p ≥ 4.
- **Escape speed does not come from the Fisher scaling.**
  - Isotropic Gaussian noise and constant-magnitude ±1 noise of the same size escape just as fast: their mean escape time is 0.74–1.05× that of Eq. (1).
  - Near the saddle the gradient is about 0, so all three are the same random walk. The measured time is close to (r / step)².
- **The Fisher scaling matters after escape.** For p ≥ 4 with the constant step:
  - Eq. (1) switches basin in 14–24.5% of trajectories, versus 98.5–100% for isotropic or sign noise;
  - it ends inside the certified region in 100%, versus 68–93%;
  - its loss floor is 6–57× lower;
  - it never diverged, while isotropic noise diverged in 26 configurations.
- **The handover radius from Thm 4 / Cor. 1 is valid and conservative.** Wherever r ≥ a_c, every Eq. (1) trajectory was inside the certified PL/KL region at its escape iteration.
- **Four statements in the paper do not hold as worded at finite η:**
  1. **"No slide-back."** 68–100% of trajectories re-enter the saddle neighbourhood after escape. This does **not** fall as η falls; it rises. At p ≥ 4 it is a short transient (all slide-backs end by iteration 1,924). At p = 2 it never stops.
  2. **"Converges to a local minimum"** with a constant step: there is a nonzero noise floor at every p (0.0009–0.108).
  3. **"Convergence rates governed by the degree p":** after handover, the rate is set by the minimum's geometry. It is −0.004004 per iteration for PL and t^(−2.08 to −2.13) for KL, the same at every p. p governs the escape time and μ, not the rate.
  4. **Eq. (35) / Eq. (41) as a noise requirement:** noise at 0.25× and 0.5× the threshold still escaped and handed over in 100% of cases. The threshold is sufficient, not necessary.

## What is analytical, numerical and empirical

- **Analytical (sympy, exact):**
  - order of the saddle: the first nonzero derivative along u = e_a has order p and value −0.1·q·p!;
  - vanishing mixed p-th derivatives;
  - stationary points {(0, 0), (±1, 0)};
  - Hessians.
- **Analytical bounds (derivation in `ablation_core.ratio_lower_bound`):** the certified regions.
  - PL: R_PL ≥ 0.2·min(p²a^(2p−2), 1).
  - KL: R_KL ≥ √(min(0.16p²a^(2p−2), 0.04/|b|) / (0.1^1.5·√2)).
- **Numerical checks (asserted in every configuration):**
  - the bound is below the exact ratio on a 1,603 × 1,201 grid inside D;
  - the Eq. (31) algebra;
  - autograd gradients of the real p-layer network equal the analytic gradient on every mini-batch, for every (p, q).

  Because every sample has the same loss (x² = x⁴ = 1), the mini-batch noise M_{n+1} ≡ 0 and **SGD is exactly gradient descent** in this model.
- **Empirical:** everything measured over 200 trajectories. A result of 100% out of 200 has a 95% Wilson interval of [98.1, 100]; it is an observed rate, not a guarantee.

## Design (fixed before any sweep)

**Model:**
- tied p-layer linear network f(x) = (aᵖx, bx);
- 4,096 inputs x ∈ {−1, +1};
- loss L_{p,q} = 0.1[(aᵖ − 1)^q + b²], with L* = 0;
- p ∈ {2, 4, 6, 8}, q ∈ {2 (PL), 4 (KL, κ = 3/4)}.

**Theory constants for every configuration:** computed exactly as Experiment 2 did (`ablation_core.theory`):
- r_lin, K′_p, K_p, K_{p+1}, and the Thm 4 constants a and b;
- the escape radius r = r_frac·r_lin;
- the noise: ρ₃ = max g_a² on |a| ≤ 0.5·r_lin, ρ₂ = ρ₃, ρ₁ = noise_mult · (ρ₂+ρ₃)·√(2K_p (0.5 r_lin)ᵖ / (η c′)), c′ = ½. The noise is always set at 0.5·r_lin, so **the radius sweep changes only the event definitions, never the dynamics.**

μ for each case:

| Case | μ from | Note |
|---|---|---|
| q = 2, p ≥ 4 | Thm 4, μ = μ_Thm4 / 2 | Thm 2 writes ‖∇L‖² ≥ 2μ(L−L*); Thm 4 writes ‖∇L‖² ≥ μ(L−L*). We use the factor-2 form. |
| q = 2, p = 2 | Thm 2, Eq. (17): μ = r²μ_min² / (2(ΔL + r²\|μ_min\|/2)) | |
| q = 4 | Cor. 1 | At p = 2 the note at the end of Sec. 4.1 is the same formula. |

**Certified region D:**
- q = 2: |a| ≥ a_c, with a_c = (min(μ, 0.2) / (0.2p²))^(1/(2p−2)).
- q = 4: |a| ≥ a_c and |b| ≤ b_c, with a_c = (μ²·0.1^1.5·√2 / (0.16p²))^(1/(2p−2)) and b_c = 0.04 / (0.16p²·a_c^(2p−2)).

**Escape bounds:** Thm 1 (p = 2) / Thm 3 (p ≥ 4), with ρ₃ evaluated on the actual escape region |a| ≤ r and the overshoot constant K = (r + max step)² − r².

**Robbins–Monro step:** n₀ = max(1000, ⌈bound⌉).

**Noises:**
- `fim`: Eq. (1) exactly as written, ±ρ₁/(ρ₂ + gᵢ²);
- `iso`: Gaussian, σ = ρ₁/ρ₂;
- `sign`: ±ρ₁/ρ₂. This is Eq. (1) without the gradient-dependent factor, so it isolates the Fisher scaling from the choice of noise distribution;
- `none`: SGD (= gradient descent).

**Schedules after the first certified handover** (Eq. (1) only):
- `const` (the paper's algorithm);
- `rm` (Robbins–Monro step, Thm 6);
- `decay`: noise × n₀/(n₀ + n − τ_h). Diagnostic, not the paper's algorithm;
- `off`: noise disabled. Diagnostic; its rate is never credited to Eq. (1).

**Sweeps** (base point: η = 0.01, r = r_lin/2, noise ×1; seed = 100q + init index everywhere, giving common random numbers; 200 trajectories per configuration):

| Sweep | What varies | Simulations | Result files |
|---|---|---|---|
| S1 degree | p × q × init {saddle, near} × {SGD, Eq.(1), iso, sign const; Eq.(1) RM, iso RM; Eq.(1) off, decay} | 128 | 128 |
| S2 step size | η ∈ {0.02, 0.005, 0.0025} × p × q × {fim, iso, sign} × variant | 216 | 216 |
| S3 noise | multiplier ∈ {0.25, 0.5, 2, 4} × p × q × {fim, iso, sign} | 96 | 96 |
| S4 radius | r_frac ∈ {0.25, 0.5, 0.75, 1.0} × p × q × {fim, iso, sign}; one simulation, four event definitions | 24 | 96 |

S2 has three variants:
- `thm`: ρ₁ recomputed at the threshold for each η, so ρ₁ ∝ η^(−1/2), with 50k iterations;
- `fixednoise`: ρ₁, ρ₂ kept at their η = 0.01 values, with 50k iterations;
- `efftime`: like `thm`, but with N·η = 500.

The S4 file at r_frac 0.5 reproduces the S1 file for the same configuration exactly.

**Events (per trajectory, fixed in code before running):**
- **Escape** τ: first |a_n| ≥ r.
- **Descent:** first L < 0.09.
- **Handover:** first entry into D. Also recorded: whether the iterate is in D at τ, and whether handover happened at or before τ.
- **Region exit:** D → not D after handover.
- **Slide-back:** **re-entry into |a| < r after escape.** This is the paper's own wording ("sliding back into unstable saddle regions"; Thm 6: "without re-entry into the saddle neighborhood").

  The task spec instead called a sign change "slide-back". That is reported separately as a **basin switch** (a changes sign after handover, i.e. it crossed the saddle line into the other minimum's basin).
- **In region at end.**
- **Divergence:** |a| > 10 or |b| > 10³.

**Statistics:**
- Escape: percent with a Wilson interval, censored count, mean over escaped runs, and the censored mean E[min(τ, N)] with a bootstrap 95% CI. Since no noisy run was censored, the censored mean equals the mean.
- Floor: median over trajectories of the mean loss over the last 10,000 iterations.
- Thm 5 trend: Cochran–Armitage test of each event proportion against ln η.

**Regression gate** (`check_regression.py`): at p = 4, η = 0.01, r_frac 0.5, noise ×1, the generalised code reproduces `../summary.json` exactly:
- every theory constant to 10⁻¹² relative;
- every summary field of all 28 Experiment-2 runs (not just the 3 required).

It also runs the p-layer torch check for all 8 (p, q).

**Pilot** (`pilot_log.txt`): p ∈ {2, 8}, η = 0.02, noise ×4, 5,000 iterations. No divergence and no implementation problems, and about 1.4 s per 5,000 iterations per core.

## Theorem verification table (`results/theorem_checks.json`)

| # | Paper statement (theorem) | Assumptions | Result | Verdict |
|---|---|---|---|---|
| 1 | Escape bound, strict saddle (Thm 1) | Eq. (8) radius; ρ₃ bounds g² on \|a\| ≤ r | 78 configs meet them, all below the bound; max measured/bound 0.55. At r = r_lin/2, Eq. (8) holds with equality (r/radius = 1.000003–1.000004, inside the 2.5·10⁻⁶ grid step); flagged as met | **Supported** |
| 2 | Escape bound, degenerate saddle (Thm 3) | Lemma 2 radius r ≤ min(r_lin, r₀); ρ₃ bound | 162 meet them, all below the bound; max ratio 0.47. The 108 that don't are also below, but no claim is made for them | **Supported** |
| 3 | Gradient information alone does not escape (Sec. 1) | M ≡ 0 here, so SGD = GD | Saddle: 0% escape at every p and q. Near start: p = 2 escapes (910 / 403 iterations); p = 4 needs 624,664 / 312,241 (exact); p = 6 about 2.1·10⁹ / 1.0·10⁹ and p = 8 about 1.0·10¹³ / 5.2·10¹² (ODE estimates) | **Supported** (exact saddle trivially; near-saddle within the budget) |
| 4 | Exit point lies in a PL/KL region (Thm 2, Thm 4, Cor. 1) | r ≥ a_c; for a PL region, μ ≤ 0.2 | 423 const-schedule configs with r ≥ a_c: Eq. (1) is in D at escape in 100% of every one. Edge cases below | **Supported where r ≥ a_c**; region statement limited for q = 2 (pitfall 1) |
| 5 | Eq. (35) / (41) noise threshold for handover | — | 17 Eq. (1) configs below threshold (16 at 0.25× / 0.5×, plus the p = 2, q = 2 base): all 100% escape and 100% handover | **Sufficient, not necessary** |
| 6 | PL needs more noise than KL (Cor. 1 discussion) | Same p and r | Theory: the Eq. (35) RHS (PL) exceeds the Eq. (41) RHS (KL) at every p: 0.560 vs 0.248, 0.150 vs 0.124, 0.0563 vs 0.0464, 0.0188 vs 0.0155. But r and μ differ between q, and μ is not comparable across q. Empirically, both reach ≥ 95% handover at the lowest multiplier tested (0.25) | **Not tested cleanly** |
| 7 | No slide-back; exp(ΔV/η) retention (Thm 5, η → 0) | Thm 5 is a limit | Re-entry rises as η falls in 6 of 8 Eq. (1) cases (in both the `thm` and the `fixednoise` variant). Basin switch falls only at p = 2 with fixed noise (2 of 8). Details below | **Not observed at η = 0.0025–0.02.** Not a contradiction of a limit theorem, but contradicts the abstract's wording |
| 8 | RM step: convergence to θ_min without re-entry w.h.p. (Thm 6) | Σα = ∞, Σα² < ∞ | RM floor is below the constant-step floor at every (p, q), e.g. 0.0034 vs 0.0284 at p = 4, q = 2. 100% in D at the end. But re-entry occurs in 69.5–98%, and basin switches in 12.5–79% (last one at iteration 19,128) | **Partly supported** |
| 9 | Linear (PL) / sublinear (KL) rates; "governed by the degree of saddle degeneracy" | Noise off after handover (diagnostic) | PL slope −0.004004 / −0.004004 / −0.004004 / −0.004009 per iteration, against predicted 2 ln(1 − 0.2η) = −0.004004. KL slopes −2.09 / −2.08 / −2.08 / −2.13, against predicted −2 | **Rates supported; dependence on p not supported** |
| 10 | Eq. (1) converges to a local minimum (p.4), constant step | — | Floors 0.105 / 0.108 (p = 2), 0.0284 / 0.0276 (p = 4), 0.0059 / 0.0053 (p = 6), 0.00105 / 0.00087 (p = 8). These are more than 10⁶× the noise-off floor | **Contradicted as worded** (constant step) |
| 11 | FIM-scaled anisotropic noise is the mechanism (Contribution 1) | — | Escape: iso/sign time ratio 0.74–1.05 (no advantage). Retention and floor: large advantage for p ≥ 4 (see summary) | **Supported for retention, not for escape** |
| 12 | Divergences and failed flags | — | 26 configurations diverged, all isotropic. Flag failures out of 536: Lemma 2 radius 144, noise threshold 181, η ≤ 1/ℓ 210, ρ₃ bound 48, r ≥ a_c 9, μ attainable 6, Eq. (8) 12; η stable at the minimum, n₀ ≥ bound and the Thm 2 gap condition never fail | See below |

## Supporting results

- **Escape bounds (A2).**
  - Measured mean escape, with 95% CI, against the bound, Eq. (1), saddle start:

    | p | q = 2 | q = 4 |
    |---|---|---|
    | 2 | 19.5 [17.5, 21.6] vs 72 | 9.2 vs 44 |
    | 4 | 35.9 vs 331 | 27.5 vs 269 |
    | 6 | 92.1 vs 864 | 88.1 vs 788 |
    | 8 | 293 vs 2,519 | 289 vs 2,451 |

  - The pure random-walk estimate (r / step)² is 10, 29, 88 and 280 for q = 2. Except at p = 2 (19.5 vs 10), the measured times are within 25% of it.
  - So escape from a flat degenerate saddle is a noise-driven random walk, as the paper's own remark on p.10 says.
  - The bounds hold with roughly 2–10× slack. Do not read anything into the bound's exponent. A2(c)–(e) show measured escape time against η, noise multiplier and radius: it moves in the same direction as the bound, with a shallower slope.
- **Handover radius (A3(c)).**
  - The paper's radius r is valid wherever r ≥ a_c: 100% of Eq. (1) trajectories are in D at the escape iteration in all 423 such configurations.
  - The radius is conservative: at r = r_lin/2 it exceeds the certified boundary by 2–13% (q = 2: 0.354/0.316, 0.420/0.374, 0.445/0.419, 0.459/0.440).
- **Rates (A4(a, b)).** With the noise switched off after certified handover:
  - PL converges geometrically at exactly the rate set by the smallest Hessian eigenvalue at the minimum (0.2, the b-direction), for every p. That is far faster than the certified PL rate ln(1 − ημ), which ranges from −8·10⁻⁴ (p = 2) to −1.3·10⁻⁶ (p = 8).
  - KL decays as t^(−2), the κ = 3/4 prediction, for every p.
- **Fisher scaling (A5b, A6, A4(c)).** The constant-magnitude sign baseline behaves like the isotropic one, not like Eq. (1). So the post-escape benefit comes from the gradient-dependent factor 1/(ρ₂ + g²), not from the shape of the noise distribution. Eq. (1)'s noise shrinks where the gradient is large (inside the basin), and the others' does not.

## Contradictions (as worded in the paper)

1. **Slide-back.**
   - **Size:** with the theory-chosen noise, 68.5–83% of Eq. (1) trajectories re-enter |a| < r after escape for p ≥ 4, and 100% for p = 2. 14–24.5% (p ≥ 4) cross the saddle into the other basin.
   - **When:** for p ≥ 4 every such event ends early. The last re-entry is at iteration 392 / 335 (p = 4), 681 / 678 (p = 6) and 1,722 / 1,449 (p = 8); the last basin switch is at iteration 1,924 or earlier. After that, 100% stay in D to iteration 50,000.
   - **Trend with η:** re-entry does not fall as η falls; it rises (e.g. p = 4, q = 2: 91, 80.5, 76, 65.5% at η = 0.0025, 0.005, 0.01, 0.02; Cochran–Armitage z = −6.2).
     - This happens even with fixed noise, the small-noise regime of Thm 5.
     - The likely reason (an interpretation, not tested separately): re-entry into the same ball one has just left is a level-crossing event. A noisy path that has just touched |a| = r recrosses it immediately with high probability, and finer steps make this more likely.
     - The basin switch, a real return through the saddle, shows no trend for p ≥ 4. It falls with η only at p = 2 with fixed noise (100% → 71% / 83.5%).
   - **Reading:** Thm 5 is an η → 0 statement, so none of this refutes it. The abstract's "remain locked … avoiding slidebacks" is not what happens at finite η.
2. **Constant-step convergence (A4(c), A6).** Eq. (1) with a constant step never converges. Its floor is set by the noise in the flat b-direction, where g ≈ 0 and the noise stays at its maximum. Robbins–Monro steps lower the floor 3.6–15× by the end of the run.
3. **"Rates governed by p" (A4(a, b)).** The asymptotic rate does not depend on p. It depends on the geometry at the minimum: the 0.2 curvature in the b-direction for PL, and the quartic order (κ = 3/4) for KL. p changes the escape time and μ, not the rate.
4. **Strict saddle, p = 2.**
   - **Never settles:** Eq. (1) with the theory-chosen noise never settles. It switches basin in 100% of trajectories up to iteration 50,000, its floor (0.105) sits at the saddle's own loss, and 90.5% (q = 2) / 98.5% (q = 4) are in D at the end.
   - **Inconsistent theory:** the Eq. (31) noise (lhs 0.300) is below the strict-saddle threshold evaluated with Thm 2's μ (0.560). Sec. 3 (Thm 2) and the Sec. 4.1 strict-saddle note give different μ at the same r.

## Limitations

- **Pitfall 1, the PL ceiling.** For q = 2 no region containing the minimum has a PL constant above 0.2, because along b the ratio is exactly 0.2.
  - At r = r_lin the theorem's μ is 0.272 at p = 8, and 0.2000003 at p = 2 (above the ceiling only by rounding).
  - D was certified with min(μ, 0.2), and these 6 configurations are flagged `flag_mu_attainable = false`.
  - So Thm 2 / Thm 4's region statement needs μ ≤ the smallest stable curvature.
- **Pitfall 2.** At r = 0.25·r_lin, q = 4, p = 4 / 6 / 8, r is below the certified a_c by 0.6–1.2%:

  | p | r | certified a_c | exact radius on b = 0 |
  |---|---|---|---|
  | 4 | 0.1685 | 0.1705 | 0.1609 |
  | 6 | 0.1922 | 0.1937 | 0.1871 |
  | 8 | 0.2052 | 0.2064 | 0.2014 |

  r is above the exact b = 0 radius, so this is slack in the √2 region bound, not a failure of Cor. 1. Eq. (1) was still in D at escape in 100% (the escape step overshoots a_c), and isotropic noise in 93.5–96.5%.
- **Pitfall 3.** For p ≥ 6 (and q = 4, p = 4 at η = 0.02), η violates η ≤ 1/ℓ on |a| ≤ 1.2, with ℓ = 69 to 13,138. Eq. (1) never diverged anyway. The 26 divergences are all isotropic Gaussian noise, mostly q = 4 with η = 0.02 or noise ×2–4. 6 of them occur where η ≤ 1/ℓ holds, so the flag alone does not explain them. The mechanism is unbounded Gaussian kicks reaching |a| > 1.2, where the q = 4 curvature is huge.
- **Radius sweep beyond r_lin/2.** Because the noise is fixed at 0.5·r_lin, the radius fractions 0.75 and 1.0 violate the Lemma 2 radius, the noise threshold and the ρ₃ bound by construction. The bounds there (10⁴–10¹⁰) are "assumption not met", even though measured escape times are far below them.
- **The model is 2-dimensional and separable.** Handover-at-escape is close to structural whenever r ≥ a_c.
- **Thm 5 and Thm 6 are limit / probability statements.** We report frequencies over 200 trajectories and at most 200k iterations.
- **Not tested:** strict saddles at p ≥ 3 with indefinite Hessians, and dimensions above 2.

## Figures (all in `figures/`, PNG + PDF)

**A1 — saddle geometry.**
- (a, b) The loss along the escape direction (b = 0) for each p. Larger p gives a flatter plateau around a = 0, i.e. a more degenerate saddle.
  - The saddle region is |a| < r (dashed lines).
  - Dots mark the certified boundary a_c. Outside it, the PL/KL inequality is proved by the analytic bound and checked on a grid.
- (c) How far the loss drops below the saddle value along u: it is 0.2aᵖ, a straight line of slope p on log-log axes. This is the definition of a p-th order saddle.
- (d) Hessian eigenvalues at the saddle: negative in a only at p = 2 (strict). For p ≥ 4 they are zero, so the Hessian cannot see the escape direction.
- (e) Noise step at zero gradient (0.11 → 0.027) against the escape radius (0.35 → 0.46). As p grows, the step shrinks and the radius grows, so escape takes more steps.
- (f) μ from the theorems against p. The q = 2 curve can pass the 0.2 PL ceiling at r = r_lin (pitfall 1).
- **Proves:** the saddle structure and where the regions are. **Cannot prove:** anything about the dynamics.

**A1b — contours and trajectories.** One Eq. (1) trajectory that stays in its basin (orange) and one that switches basin (green), for each (p, q), first 1,500 iterations.
- p = 2 has no "stays" example, because every trajectory switches. Its |b| exceeds 1 (clipped in the plot).
- **Illustrative only:** two trajectories out of 200.

**A2 — empirical vs theoretical escape.**
- (a) Every configuration: measured mean against its bound on log-log axes. The line is y = x; filled markers have the assumptions met, hollow markers don't. All points lie below the line.
- (b) Escape time against p, with the bound, the (r/step)² random walk, and SGD from the near start. SGD's times beyond 2·10⁶ are ODE estimates.
- (c–e) Against η, noise multiplier and radius. In (e) the trajectories are identical; only the event definition changes.
- (f) Measured/bound, assumptions met: between 0.076 and 0.55.
- **Proves:** the bound holds in these configurations. **Cannot prove:** that the bound's exponent is right (the bound is not tight).

**A3 — success rates.** Escape % by optimizer × start × p, descent %, in-region-at-escape % against radius, and escape % against noise multiplier. Almost every cell is 100%; that is itself the result.
- SGD: 0% at the saddle; 0% from the near start for p ≥ 4.
- Eq. (1) still escapes at 0.25× the threshold noise.

**A4 — convergence.**
- (a, b) The diagnostic noise-off rate after handover: PL geometric at −0.004 for every p, KL t^(−2) for every p.
- (c) Floor by p and schedule. Constant-step Eq. (1) has a floor; isotropic noise has a much higher one.
- (d, e) Schedule comparison at p = 4.
- (f) Floor against η, nearly flat because the theorem noise rescales with η.
- **Proves:** the region's rate (PL/KL), and that the floor is persistent. **Cannot prove:** a rate for Eq. (1) itself, which has a floor.

**A5 / A5b — slide-back heatmaps.** p × η and p × noise multiplier, for Eq. (1) and isotropic noise, q = 2 and q = 4. A5 shows the paper's slide-back (saddle re-entry); A5b shows basin switch. The numbers are % of 200 trajectories.
- **Proves:** slide-backs are frequent at these η and do not fall as η falls. **Cannot prove:** anything about the η → 0 limit.

**A6 — after-handover schedules.** Slide-back %, in-region-at-end % and final loss for constant / RM / decay / off, at every (p, q). Decay and off are diagnostics, not the paper's algorithm.

## Suggested wording changes for the paper

1. **Abstract and Sec. 5.** Replace "remain locked inside the local minimum basin … avoiding slide-backs" with "after a short transient, remain in the local minimum basin; slide-backs become exponentially rare as η → 0 (Thm 5)". Define slide-back as re-entry into a strictly smaller neighbourhood (e.g. B(θ_deg, r/2)) or as a basin change, not as re-entry into the exit ball itself.
2. **p.4.** "Converges to a local minimum" → "converges to a neighbourhood of a local minimum with a constant step size; converges with Robbins–Monro step sizes (Thm 6)".
3. **Abstract** ("exhibiting convergence rates governed by the degree of saddle degeneracy"). Say instead that p governs the **escape time and the handover radius / constant**, while the post-handover rate is set by the destination's PL/KL geometry.
4. **Eq. (35) / (41).** Call them sufficient conditions. The experiments hand over at 0.25× the threshold.
5. **Contribution 1** ("directional amplifier"). The ablation shows the gradient-dependent scaling does not speed up escape (isotropic noise is as fast). Its benefit is after escape: lower slide-back, staying in the region, a lower floor, and no divergence.
6. **Thm 2 / Thm 4 region statement.** Add that μ cannot exceed the smallest curvature of the stable directions (here 0.2); otherwise the inequality holds only pointwise along the escape axis.
7. **Thm 2 vs the Sec. 4.1 strict-saddle note.** Reconcile μ: at the same r they give 0.08 (Thm 2) against the value implied by the Eq. (35) form. Also, the Eq. (8) plug-in uses r = κ|μ_min|/L₂, while Eq. (8) itself requires r < κ|μ_min|/(2L₂).
