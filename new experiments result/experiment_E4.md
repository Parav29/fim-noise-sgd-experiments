# Experiment E4 — no slide-back after escape (constant and Robbins–Monro steps)

Figure: `fig_E4_no_slideback.png`. Common setup and definitions: `README.md`.

## What it tests
The draft's retention claims:
- **Thm 5** (p.15): with a constant step and η → 0, the iterate stays in the basin for time ≳ exp((ΔV − ε)/η).
- **Thm 6** (p.16): with a Robbins–Monro step-size sequence (Σα_n = ∞, Σα_n² < ∞), after exit the probability of converging without re-entering the saddle neighbourhood is ≥ 1 − O(tail sum of α_n²).

## What exactly is run
Same setting as E1 (θ₀ = 0.1 × default init, ρ₁ = 10⁻⁷, ρ₂ = 10⁻⁶), two step rules for Eq. (1), plus SGD under the decaying rule:
- constant step α_n = η = 0.01 (these are the E1 runs);
- Robbins–Monro step α_n = η·n₀/(n₀ + n), n₀ = 2,000 (so α_n ~ 1/n: Σα_n diverges, Σα_n² converges). The step multiplies both the gradient and the noise, as in Eq. (1).
- 3 seeds each, 19,500 iterations. A **slide-back** is a return of the full training loss to ≥ 2.2026 at any evaluation after the escape iteration.

## Assumptions
- n₀ = 2,000 was fixed in advance and not tuned.
- The plateau band (loss ≥ 2.2026) stands in for "the saddle neighbourhood". The paper's neighbourhood is a ball of radius r around θ_deg in parameter space; the loss band is a proxy.

## Results
| Step rule | Seed | Escape iteration | Returned to plateau band? | Largest rise above the running minimum after escape | Final loss |
|---|---|---|---|---|---|
| constant | 0 / 1 / 2 | 1,500 / 1,500 / 1,500 | no / no / no | 0.034 / 0.034 / 0.035 | 0.959 / 0.951 / 0.944 |
| Robbins–Monro | 0 / 1 / 2 | 2,600 / 2,500 / 2,600 | no / no / no | 0.003 / 0.001 / 0.004 | 1.501 / 1.500 / 1.483 |

- 0 of 6 Eq. (1) runs slid back. SGD under the decaying step also stays at 2.3026 throughout (same stall as E1).
- Escape under the decaying step is later (about 2,550 vs 1,500 iterations) because the noise-driven steps shrink as n grows. Final loss is higher because the step is about 10× smaller by the end.
- After escape the loss curve is nearly monotone: the largest temporary rise is 0.035 (constant) and 0.004 (Robbins–Monro).

## What this shows
Under both step rules, once Eq. (1) escapes, the loss never returns to the plateau within 19,500 iterations, and escape still occurs when the step size decays, which is the Robbins–Monro setting of Thm 6. The decaying step's lower noise fluctuation is also visible (rise 0.004 vs 0.034).

## What this does not show
- Thm 5's exponentially long retention time or Thm 6's probability bound: we observe 6 runs with no slide-back, not a probability estimate and not an exponential horizon.
- That the iterate is in a PL/KL region or near a minimum: the loss is still decreasing at the end of every run.
- A parameter-space check of the saddle neighbourhood; only the loss band was used.
