# Experiment E3 — escape time vs noise level

Figure: `fig_E3_noise_ratio.png`. Common setup and definitions: `README.md`.

## What it tests
Two statements in the draft:
1. **Escape time bound** (Thm 1, p.7; Thm 3, p.12): E[τ] ≤ N + c·((ρ₂ + ρ₃)/(η ρ₁))²·(r² + K). The bound decreases as ρ₁ grows (as ρ₁⁻²).
2. **Noise-to-step-size level** (p.4; Eq. (35), p.12; "Trade-off" paragraph, p.15): entering a PL region requires η(ρ₁/(ρ₂+ρ₃))² to exceed a threshold.

## What exactly is run
Same setting as E1 (θ₀ = 0.1 × default init, η = 0.01, ρ₂ = 10⁻⁶, constant step, 19,500 iterations), Eq. (1) only, with four noise levels:

| ρ₁ | ρ₁/ρ₂ | η(ρ₁/ρ₂)² |
|---|---|---|
| 1·10⁻⁸ | 0.01 | 1·10⁻⁶ |
| 3·10⁻⁸ | 0.03 | 9·10⁻⁶ |
| 1·10⁻⁷ | 0.1 | 1·10⁻⁴ (these are the E1 runs) |
| 3·10⁻⁷ | 0.3 | 9·10⁻⁴ |

3 seeds per level. The plateau baseline is the SGD run from E1. The x-axis of the right panel uses η(ρ₁/ρ₂)², which is Eq. (35) with ρ₃ → 0 (ρ₃ is a bound on g², which is tiny at this start).

## Assumptions
- Only ρ₁ was varied. η and the init scale were held fixed, so the η-dependence of the bound is not tested.
- ρ₃ is neglected in the x-axis. The constants r, K, c of the bound are unknown, so absolute bound values cannot be compared, only the dependence on ρ₁.
- The 19,500-iteration budget caps the longest observable escape time.

## Results
| ρ₁ | Escape iterations (seeds 0, 1, 2) | Mean | Final loss (seeds 0, 1, 2) |
|---|---|---|---|
| 1·10⁻⁸ | 12,700 / 13,500 / 13,600 | 13,267 | 1.531 / 1.539 / 1.510 |
| 3·10⁻⁸ | 4,800 / 5,400 / 4,900 | 5,033 | 1.169 / 1.149 / 1.145 |
| 1·10⁻⁷ | 1,500 / 1,500 / 1,500 | 1,500 | 0.959 / 0.951 / 0.944 |
| 3·10⁻⁷ | 500 / 500 / 600 | 533 | 0.950 / 0.941 / 0.922 |

- **All 12 runs escape** within the budget. Escape time falls monotonically as ρ₁ grows, with very little seed-to-seed spread.
- **Measured exponent:** a straight-line fit of ln(escape iteration) against ln ρ₁ over all 12 runs gives slope **−0.95** (local slopes between neighbouring levels: −0.88, −1.01, −0.94).
- Final loss is also ordered by ρ₁ because a longer escape leaves less time to train.

## What this shows
- The direction predicted by the bound holds: more noise (larger η ρ₁) means faster escape, over a range of 30× in ρ₁ and about 25× in escape time.
- The measured dependence is roughly ρ₁⁻¹, shallower than the ρ₁⁻² of the bound's form. This is not a contradiction, because the paper states an **upper** bound with unknown constants; it only means the bound is not tight in this network. We have no verified explanation for why the exponent is about −1.
- There is no sharp threshold in this range: the "larger noise-to-step-size ratio" statement shows up as a smooth slowdown. At ρ₁ = 10⁻⁸ escape takes 13k of the 19.5k available iterations, so with a shorter budget this level would look like a failure to escape.

## What this does not show
- It does not test whether the exit point is in a PL region (Eq. (35) is about PL entry, not escape). We measure escape only.
- Levels below ρ₁ = 10⁻⁸ were not run, so whether a true no-escape regime exists, and where, is unknown.
- The η-dependence of the bound (η⁻²) was not tested.
- The earlier max-pool pilot (4,000 iterations) found ρ₁/ρ₂ = 0.01 "stuck"; this run shows it does escape, at about 13k iterations. The pilot's budget was too short to see it.
