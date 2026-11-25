# Clean figures for the paper's Experiments section

Publication versions of the figures for the two final experiments. **Nothing was re-run.** `make_clean_figures.py` only reads the existing results in place:
- `../final_experiment_1_saddle_escape/raw/E1_smallinit_seed{0,1,2}.json`
- `../final_experiment_2_handover_convergence/summary.json` and `raw/*.npz`

The original figures in those folders are unchanged.

Regenerate: `python3 make_clean_figures.py` (about 20 s). Every figure is written as `.pdf` (vector, for LaTeX) and `.png` (300 dpi). Width is 7 in (full text width). The fonts are STIX serif, to match a LaTeX body font.

| File | Experiment | Panels |
|---|---|---|
| `fig1_cifar_saddle_escape` | Exp. 1 | (a) training loss, (b) log training loss, (c)/(d) 3D loss landscape around the final SGD / Eq. (1) weights |
| `fig2_handover_geometry` | Exp. 2 | (a)/(c) 3D loss surface with a trajectory, PL / KL; (b)/(d) contour plot with trajectories, certified region, escape radius, escape and handover points |
| `fig3_escape_and_certified_handover` | Exp. 2 | (a)/(b) loss gap vs. iteration, PL / KL; (c)/(d) measured ratio and certified lower bound ψ vs. μ |
| `fig4_retention_and_convergence` | Exp. 2 | (a) distances and the two radii (PL); (b) slide-back and settling rates; (c) PL convergence after handover; (d) KL convergence and noise floor vs. accumulated step |

Changes from the working figures:
- **Fewer panels per figure.** The Exp. 2 material is now three figures instead of one 8-panel sheet per configuration, and PL/KL sit side by side wherever they are comparable.
- **One summary per curve.** Each curve is a median with an interquartile band (the working figures showed mean, median and a 5–95% band together).
- **No suptitles.** Setup details move to the captions.
- **Shared legends, and nothing drawn on top of the data.**
- **No clipping.** The 3D surfaces in Fig. 2 use a domain where L ≤ 0.2, so nothing is cut off.
- **Clearer trajectory examples.** The example trajectories are one that stays in its basin and one that slides back through the saddle. For PL, none of the 20 stored trajectories ends in the other basin, so the PL example crosses the saddle and then returns.

## Draft captions (numbers from `summary.json` / `convergence_numbers.json`)

**Figure 1.** Noise-assisted escape on CIFAR-10 (66,570-parameter GeLU CNN with average pooling, cross-entropy). Setup: θ₀ = 0.1 × default initialisation, η = 0.01, ρ₁ = 10⁻⁷, ρ₂ = 10⁻⁶, batch 128, 50 epochs; mean of 3 seeds, band = min–max.
- (a) Full training loss. SGD remains at ln 10 = 2.3026 for all 19,500 iterations (maximum deviation 7.6·10⁻⁶). Eq. (1) leaves the plateau at iteration 1,500 on every seed and reaches 0.951.
- (b) Same, on a log scale.
- (c, d) Loss on 5,000 training images along two filter-normalised random directions around the final weights (seed 0). The directions are scaled to each run's weights, so the panels illustrate rather than compare. They do not establish PL membership or the order of the saddle.

**Figure 2.** Handover geometry in the tied four-layer linear network f(x) = (a⁴x, bx) with L_q = 0.1[(a⁴ − 1)^q + b²]. There is a fourth-order degenerate saddle at (0, 0) and minima at (±1, 0).
- (a, c) Loss surface and one Eq. (1) trajectory (first 1,500 iterations).
- (b, d) Contours with two trajectories. Solid red: boundary of the certified region D (|a| ≥ a_c = 0.374 for PL, 0.317 for KL; on D, R_PL ≥ μ = 0.0087 and R_KL ≥ μ = 0.240). Dashed: the escape radius r = 0.420 / 0.337 from Thm 4 / Cor. 1.
- Because r > a_c, every escape lands in D. SGD started at the saddle never moves.

**Figure 3.** Escape and certified handover (200 trajectories per curve; median and interquartile range).
- (a, b) Loss gap L − L* (L* = 0). Vertical lines: median handover iteration τ_h and escape iteration τ_esc (19 / 24 for PL, 17 / 21 for KL). Eq. (1) and isotropic noise escape at the same rate. SGD stays at the saddle loss. With a constant step, Eq. (1) settles at a floor of about 0.028; the dashed curve switches the noise off after handover (a diagnostic, not the paper's algorithm).
- (c, d) Measured PL/KL ratio and its certified pointwise lower bound ψ. The bound crosses μ (dotted) at handover. The measured ratio can exceed μ earlier, which is why membership is decided by ψ, not by the measured ratio.

**Figure 4.** Retention and convergence after handover.
- (a) PL: |a_n| for Eq. (1) and isotropic noise, and the distance of Eq. (1) to its destination minimum (median, IQR). The certified boundary a_c and the escape radius r are marked.
- (b) Share of 200 trajectories that slide back through the saddle after handover, and that have no region exit after n = 5,000 (solid PL, hatched KL). Eq. (1): 21–25% slide back, all within the first 430 iterations; afterwards 100% stay in D. Isotropic noise with a constant step never settles.
- (c) PL, iterations since handover. With the noise switched off, the loss decays geometrically at the local rate (1 − ηλ_min)^{2k}, λ_min = 0.2 (fitted slope −0.004004 per iteration). This is faster than the certified rate (1 − ημ)^k. With a constant step, Eq. (1) stays at a noise floor; the Robbins–Monro step keeps decreasing.
- (d) KL, against the accumulated step t = Σα_n. With the noise off, the decay follows the KL(κ = 3/4) prediction t^{−1/(2κ−1)} = t^{−2} (fitted slope −2.08). The constant-step noise floor is 0.027 for Eq. (1) and 0.17 for isotropic noise.
