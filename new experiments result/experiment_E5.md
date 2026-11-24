# Experiment E5 — loss decay after escape (linear vs sublinear)

Figure: `fig_E5_post_escape_rate.png`. Common setup and definitions: `README.md`. This is an analysis of the E1 and E4 runs, with no new training.

## What it tests
The draft states (p.2, p.4, p.14) that convergence inside a PL region is linear and inside a KL region is sublinear, and that the handover leads to one of them. If the loss gap decays linearly, ln(L − L\*) is a straight line in n; a sublinear rate bends it.

## What exactly is done
For Eq. (1) runs (constant step: E1, Robbins–Monro step: E4), each seed, align the curves at the escape iteration τ and fit the least-squares slope of ln L(θ_n) in windows of iterations since escape. L\* is unknown, so L_ref = 0 and the plotted quantity is ln L, not ln(L − L\*).

## Assumptions
- L\* = 0 is used because it is the infimum of cross-entropy. The true L\* of this network and data is positive and unknown, so the true gap is smaller than L and its log-slope is steeper than shown.
- Windows are measured from the escape iteration; the last window ends at the end of the run (18,000 iterations after escape for the constant step; about 17,000 for the Robbins–Monro step).

## Results (slope of ln L per 1,000 iterations since escape; seeds 0 / 1 / 2)
| Window since escape | constant step | Robbins–Monro step |
|---|---|---|
| 0–1,500 | −0.146 / −0.148 / −0.156 | −0.053 / −0.061 / −0.062 |
| 1,500–4,000 | −0.084 / −0.074 / −0.080 | −0.036 / −0.036 / −0.037 |
| 4,000–9,000 | −0.034 / −0.036 / −0.037 | −0.023 / −0.022 / −0.026 |
| 9,000–14,000 | −0.025 / −0.027 / −0.026 | −0.013 / −0.014 / −0.012 |
| last window | −0.023 / −0.023 / −0.021 | −0.007 / −0.008 / −0.006 |

- **Constant step:** the log-slope shrinks from −0.15 to about −0.025 and then changes little over the last two windows (−0.026 → −0.022). The curve is not a straight line early on; it flattens towards a slow, nearly constant slope.
- **Robbins–Monro step:** the slope keeps shrinking to −0.007. Since the step itself shrinks as 1/n, a decaying slope is expected even for a perfectly linear rate, so this curve carries no information about PL vs KL.
- The three seeds agree closely in every window.

## What this shows
For the constant-step runs the post-escape loss decays quickly at first and then settles into a slow, nearly log-linear tail. This is *compatible* with "sublinear progress turning into a linear regime", and also with sublinear decay that has not yet been resolved.

## What this does not show
- It does not distinguish PL (linear) from KL (sublinear): L\* is unknown (the slope depends on it), the last two windows differ slightly (−0.026 vs −0.022) so "constant" is not established, and the loss is still far from any minimum (about 0.95).
- It is not evidence of a PL region, and the text of the paper should not cite it as such. It should be described as "the post-escape rate", with this caveat.
