"""Final Experiment 2: degenerate saddle -> noise-assisted escape -> certified PL/KL region -> retention -> convergence.

Model : tied 4-layer deep linear network f_{a,b}(x) = (a^4 x, b x) (identity activations), targets y = (x, 0).
Data  : 4096 inputs x ~ Uniform{-1,+1}.
Loss  : per-sample l = 0.1[(f_1 - y_1)^q + (f_2 - y_2)^2]  =>  L_q(a,b) = 0.1[(a^4-1)^q + b^2]  (x^2 = x^4 = 1).
        q = 2: PL destination;  q = 4: KL destination with exponent kappa = 3/4.
All theory constants are computed from the paper's formulas (Lemma 2 / Eq. 31, Thm 3, Thm 4, Cor. 1, Eq. 35/41)
and checked before any training.  Run:  python3 run_experiment2.py   (about 2-4 min on a laptop CPU)
"""
import json, math, os
import numpy as np
import sympy as sp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
ETA, CPRIME, N_ITER, N_TRAJ, N0_RM, P, KAPPA = 0.01, 0.5, 50_000, 200, 1000, 4, 0.75
INITS = {"saddle": (0.0, 0.0), "near": (0.01, 0.01)}
N_KEEP = 20  # trajectories stored in full (at grid resolution) for the landscape/contour panels


# ----------------------------------------------------------------------------- analytic model
def loss(q, a, b):
    return 0.1 * ((a ** 4 - 1) ** q + b ** 2)


def grad(q, a, b):
    return 0.1 * q * (a ** 4 - 1) ** (q - 1) * 4 * a ** 3, 0.2 * b


def ratio(q, a, b):
    """Measured R_PL = |grad|^2 / (2(L-L*)) for q=2, R_KL = |grad| / (L-L*)^kappa for q=4 (L* = 0)."""
    ga, gb = grad(q, a, b)
    g2, L = ga ** 2 + gb ** 2, loss(q, a, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        return g2 / (2 * L) if q == 2 else np.sqrt(g2) / L ** KAPPA


def ratio_lower_bound(q, a, b):
    """Rigorous pointwise lower bound psi(a,b) <= ratio(a,b) (derivation in README.md, 'Certified regions').
    q=2: R_PL = (3.2 a^6 A + 0.2 B)/(A+B) with A=(a^4-1)^2, B=b^2  >= min(3.2 a^6, 0.2).
    q=4: R_KL^2 = (2.56 a^6 A^1.5 + 0.04 B)/(0.1^1.5 (A+B)^1.5), A=(a^4-1)^4; (A+B)^1.5 <= sqrt2 (A^1.5+B^1.5)
         and 0.04 B >= (0.04/|b|) B^1.5  =>  R_KL >= sqrt(min(2.56 a^6, 0.04/|b|) / (0.1^1.5 sqrt2))."""
    if q == 2:
        return np.minimum(3.2 * a ** 6, 0.2)
    with np.errstate(divide="ignore"):
        m = np.minimum(2.56 * a ** 6, 0.04 / np.abs(b))
    return np.sqrt(m / (0.1 ** 1.5 * math.sqrt(2)))


# ----------------------------------------------------------------------------- theory (before training)
def theory(q):
    A, B, T = sp.symbols("a b t", real=True)
    Ls = sp.Rational(1, 10) * ((A ** 4 - 1) ** q + B ** 2)
    du = [sp.diff(Ls.subs({A: T, B: 0}), T, k).subs(T, 0) for k in range(1, P + 1)]
    mixed = [sp.diff(Ls, A, k, B, P - k).subs({A: 0, B: 0}) for k in range(P)]
    stat = sp.solve([sp.diff(Ls, A), sp.diff(Ls, B)], [A, B], dict=True)
    hess0 = sp.hessian(Ls, (A, B)).subs({A: 0, B: 0})
    hess1 = sp.hessian(Ls, (A, B)).subs({A: 1, B: 0})
    assert du[:P - 1] == [0] * (P - 1) and du[P - 1] < 0, "saddle is not of order p=4 along u=e_a"
    assert all(m == 0 for m in mixed), "mixed p-th derivatives must vanish (L_p = 0)"
    assert len(stat) == 3 and {(s[A], s[B]) for s in stat} == {(0, 0), (1, 0), (-1, 0)}
    mu_deg = float(du[P - 1])
    lead = abs(mu_deg) / math.factorial(P - 1)       # |mu_deg|/(p-1)!: grad_a = -lead a^3 + R_p(a)
    ag = np.linspace(1e-6, 1.0, 400_001)
    Rp = np.abs(grad(q, ag, 0.0)[0] + lead * ag ** 3)  # gradient Taylor remainder along u
    bad = Rp > 0.5 * lead * ag ** 3                   # r_lin: leading term dominates remainder by 2x
    r_lin = float(ag[np.argmax(bad)] if bad.any() else ag[-1])
    m = ag <= r_lin
    Kp_prime = float(np.max(Rp[m] / ag[m] ** P))      # |R_p| <= K'_p |Z|^p on |Z| <= r_lin
    Kp = lead + Kp_prime * r_lin                      # Eq. (31) with L_p = 0, C_0 = 0
    Kp1 = float(np.max(np.abs(loss(q, ag[m], 0) - 0.1 - mu_deg / 24 * ag[m] ** P) / ag[m] ** P))
    opnorm = abs(mu_deg) / math.factorial(P)          # ||grad^p L|| / p!  (only the a^4 component)
    dL = 0.1                                          # Delta L = L(theta_deg) - L*

    r = r_lin / 2                                     # design choice: handover/escape radius = r_lin/2
    a_c4 = (lead - Kp_prime * r) ** 2                 # Thm 4 constant 'a' with R(r0)=1
    b_c4 = r ** P * (opnorm + Kp1)                    # Thm 4 constant 'b'
    if q == 2:   # Thm 4 (convention |grad|^2 >= mu_T4 (L-L*)): r^{2(p-1)} a >= mu_T4 (dL + b)
        mu_T4 = r ** (2 * (P - 1)) * a_c4 / (dL + b_c4)
        mu = mu_T4 / 2                                # R_PL convention |grad|^2 >= 2 mu (L-L*)
        a_crit = (mu / 3.2) ** (1 / 6)                # exact certified boundary
        b_crit = math.inf                             # PL bound holds for every b
        exact_r_b0 = a_crit
        rhs35 = 2 * Kp / CPRIME * (mu_T4 * (dL + b_c4) / a_c4) ** (P / (2 * (P - 1)))      # Eq. (35)
    else:        # Cor. 1: r^{p-1} sqrt(a) >= mu (dL + b)^kappa
        mu = r ** (P - 1) * math.sqrt(a_c4) / (dL + b_c4) ** KAPPA
        a_crit = (mu ** 2 * 0.1 ** 1.5 * math.sqrt(2) / 2.56) ** (1 / 6)
        b_crit = 0.04 / (2.56 * a_crit ** 6)          # largest |b| keeping the a-term binding
        exact_r_b0 = (mu * 0.1 ** KAPPA / 1.6) ** (1 / 3)
        rhs35 = 2 * Kp / CPRIME * mu ** (P / (P - 1)) * (dL + b_c4) ** (KAPPA * P / (P - 1)) / a_c4 ** (P / (2 * (P - 1)))  # Eq. (41)
    inf_bound = float(ratio_lower_bound(q, np.array(a_crit), np.array(0.0 if q == 2 else b_crit)))
    # independent numerical check of the certified region: exact ratio on a dense grid inside D
    ga_ = np.concatenate([-np.geomspace(a_crit, 1.6, 801), np.geomspace(a_crit, 1.6, 801)])
    bmax = 10.0 if q == 2 else b_crit
    gb_ = np.concatenate([-np.geomspace(1e-8, bmax, 600), [0.0], np.geomspace(1e-8, bmax, 600)])
    AA, BB = np.meshgrid(ga_, gb_)
    keep = loss(q, AA, BB) > 1e-14                    # exclude the minimum itself (0/0)
    grid_inf = float(np.min(ratio(q, AA, BB)[keep]))
    assert inf_bound >= mu * (1 - 1e-9) and grid_inf >= inf_bound * (1 - 1e-9), (mu, inf_bound, grid_inf)
    assert r >= a_crit, "paper radius must not be inside the certified boundary"

    rho3 = float(np.max(grad(q, np.linspace(0, r, 100_001), 0.0)[0] ** 2))  # g_a^2 <= rho3 on |Z| <= r
    rho2 = rho3                                       # design choice: noise halves where g^2 = rho3
    rho1 = (rho2 + rho3) * math.sqrt(2 * Kp * r ** P / (ETA * CPRIME))  # Eq. (31) radius = r exactly
    r0 = (ETA * CPRIME * rho1 ** 2 / (2 * Kp * (rho2 + rho3) ** 2)) ** (1 / P)
    lhs35 = ETA * (rho1 / (rho2 + rho3)) ** 2
    assert abs(r0 - r) < 1e-12 and r0 <= r_lin and abs(lhs35 / rhs35 - 1) < 1e-9

    aa = np.linspace(-1.2, 1.2, 240_001)              # smoothness: eta <= 1/ell on |a| <= 1.2
    h = aa[1] - aa[0]
    ell = float(max(np.max(np.abs(np.gradient(grad(q, aa, 0.0)[0], h))), 0.2))
    assert ETA <= 1 / ell, (ETA, 1 / ell)
    step_max = ETA * (math.sqrt(rho3) + rho1 / rho2)  # largest |Delta a| per step inside the r-slab
    K_over = (r + step_max) ** 2 - r ** 2             # overshoot constant K in Thm 3
    tau_bound = 1 / (1 - CPRIME) * ((rho2 + rho3) / (ETA * rho1)) ** 2 * (r ** 2 + K_over)  # Thm 3, N=0
    assert N0_RM >= tau_bound

    a0 = INITS["near"][0]                             # deterministic GD escape time from the near-saddle init
    x, n = a0, 0
    while abs(x) < r and n < 5_000_000:
        x -= ETA * grad(q, x, 0.0)[0]; n += 1
    return dict(q=q, p=P, mu_deg=mu_deg, lead=lead, hessian_at_saddle=str(hess0.tolist()),
                hessian_at_min=str(hess1.tolist()), stationary_points=str([(s[A], s[B]) for s in stat]),
                r_lin=r_lin, Kp_prime=Kp_prime, Kp=Kp, Kp1=Kp1, opnorm_over_pfact=opnorm, dL=dL,
                r=r, thm4_a=a_c4, thm4_b=b_c4, mu=mu, mu_convention=("R_PL=|g|^2/(2(L-L*))" if q == 2 else
                "R_KL=|g|/(L-L*)^0.75"), a_crit=a_crit, b_crit=b_crit, exact_radius_along_b0=exact_r_b0,
                certified_inf_bound=inf_bound, grid_inf_ratio_in_D=grid_inf, rho1=rho1, rho2=rho2, rho3=rho3,
                r0_eq31=r0, eq35_lhs=lhs35, eq35_rhs=rhs35, eta=ETA, c_prime=CPRIME, smoothness_ell=ell,
                thm3_overshoot_K=K_over, thm3_tau_bound=tau_bound, n0_rm=N0_RM, iso_sigma=rho1 / rho2,
                gd_escape_iters_from_near_init=(n if abs(x) >= r else None))


def torch_check(q):
    """The network as an actual 4-layer tied linear net on the 4096-point dataset: autograd mini-batch gradients
    equal the analytic full gradient, and every mini-batch gives the same gradient (so M_{n+1} == 0)."""
    g = torch.Generator().manual_seed(0)
    X = (torch.randint(0, 2, (4096,), generator=g) * 2 - 1).double()
    Y = torch.stack([X, torch.zeros_like(X)], 1)
    for a0, b0 in [(0.3, -0.7), (1.1, 0.2), (-0.6, 1.5)]:
        grads = []
        for idx in torch.randperm(4096, generator=g)[:384].view(3, 128):
            a, b = torch.tensor(a0, dtype=torch.float64, requires_grad=True), torch.tensor(b0, dtype=torch.float64, requires_grad=True)
            h = X[idx]
            for _ in range(4):               # four tied weight layers, identity activation
                h = a * h
            out = torch.stack([h, b * X[idx]], 1)
            l = (0.1 * ((out[:, 0] - Y[idx, 0]) ** q + (out[:, 1] - Y[idx, 1]) ** 2)).mean()
            l.backward()
            grads.append([a.grad.item(), b.grad.item()])
        ga, gb = grad(q, a0, b0)
        assert np.allclose(grads, [[ga, gb]] * 3, rtol=1e-12, atol=1e-12)
    return True


# ----------------------------------------------------------------------------- simulation
def grid(N):
    return np.unique(np.concatenate([np.arange(0, 500), np.arange(500, 5000, 10), np.arange(5000, N + 1, 100), [N]]))


def simulate(q, th, init, noise, step, noise_off_after_handover=False, seed=0):
    S, N = N_TRAJ, N_ITER
    rng = np.random.default_rng(seed)
    a, b = np.full(S, init[0]), np.full(S, init[1])
    r, ac, bc, rho1, rho2 = th["r"], th["a_crit"], th["b_crit"], th["rho1"], th["rho2"]
    sig = rho1 / rho2
    G = grid(N)
    gi = np.full(N + 1, -1)
    gi[G] = np.arange(len(G))
    Q = [5, 25, 50, 75, 95]
    stats = {k: np.full((len(G), len(Q) + 1), np.nan) for k in
             ("loss", "ratio", "ratio_lb", "dist_saddle", "dist_min", "noise_step", "abs_a", "abs_b")}
    keep_a = np.zeros((len(G), N_KEEP), np.float32)
    keep_b = np.zeros((len(G), N_KEEP), np.float32)
    post = np.full((S, len(G)), np.nan, np.float32)   # L at n - tau_handover = G[k]
    t_esc = np.full(S, -1); t_h = np.full(S, -1); t_div = np.full(S, -1)
    inD_at_esc = np.zeros(S, bool); sign_h = np.zeros(S)
    exits = np.zeros(S, int); last_exit = np.full(S, -1); reentry = np.zeros(S, int); switches = np.zeros(S, int)
    alive = np.ones(S, bool)
    prev_inD = np.zeros(S, bool); prev_sticky = np.ones(S, bool); prev_sign = np.sign(a)
    tail = np.zeros(S); tail_n = 0
    for n in range(N + 1):
        absa = np.abs(a)
        inD = (absa >= ac) & (np.abs(b) <= bc) & alive
        sticky = (absa < r) & alive
        new_e = (t_esc < 0) & ~sticky & alive
        t_esc[new_e] = n; inD_at_esc[new_e] = inD[new_e]
        new_h = (t_h < 0) & inD
        t_h[new_h] = n; sign_h[new_h] = np.sign(a[new_h])
        after_h = (t_h >= 0) & (t_h < n) & alive
        ex = after_h & prev_inD & ~inD
        exits += ex; last_exit[ex] = n
        re = (t_esc >= 0) & (t_esc < n) & alive & ~prev_sticky & sticky
        reentry += re
        sg = np.sign(a)
        switches += after_h & (sg != prev_sign) & (sg != 0) & (prev_sign != 0)
        prev_inD, prev_sticky = inD, sticky
        prev_sign = np.where(sg != 0, sg, prev_sign)
        L = loss(q, a, b)
        k = n - t_h
        ok = (t_h >= 0) & alive
        kk = gi[np.clip(k, 0, N)]
        ok &= kk >= 0
        post[ok, kk[ok]] = L[ok]
        if n >= N - 10_000:
            tail += L; tail_n += 1
        ga, gb = grad(q, a, b)
        if noise == "fim":
            s = rng.integers(0, 2, (2, S)) * 2 - 1
            ua, ub = s[0] * rho1 / (rho2 + ga ** 2), s[1] * rho1 / (rho2 + gb ** 2)
        elif noise == "iso":
            ua, ub = rng.normal(0.0, sig, (2, S))
        else:
            ua, ub = np.zeros(S), np.zeros(S)
        if noise_off_after_handover:
            off = t_h >= 0
            ua, ub = np.where(off, 0.0, ua), np.where(off, 0.0, ub)
        al = ETA if step == "const" else ETA * N0_RM / (N0_RM + n)
        if gi[n] >= 0:
            j = gi[n]
            dest = np.where(t_h >= 0, sign_h, 1.0)
            dmin = np.where(t_h >= 0, np.hypot(a - dest, b), np.hypot(absa - 1, b))
            vals = dict(loss=L, ratio=ratio(q, a, b), ratio_lb=ratio_lower_bound(q, a, b),
                        dist_saddle=np.hypot(a, b), dist_min=dmin, noise_step=al * np.hypot(ua, ub), abs_a=absa,
                        abs_b=np.abs(b))
            for key, v in vals.items():
                v = np.where(alive, v, np.nan)
                if np.isfinite(v).any():
                    stats[key][j, :len(Q)] = np.nanpercentile(v, Q)
                    stats[key][j, -1] = np.nanmean(v)
            keep_a[j], keep_b[j] = a[:N_KEEP], b[:N_KEEP]
        if n == N:
            break
        with np.errstate(over="ignore", invalid="ignore"):
            a = a - al * (ga + ua)
            b = b - al * (gb + ub)
        div = alive & (~np.isfinite(a) | ~np.isfinite(b) | (np.abs(a) > 10) | (np.abs(b) > 1e3))
        t_div[div] = n + 1
        alive &= ~div
        a = np.where(alive, a, np.nan); b = np.where(alive, b, np.nan)
    import warnings
    with warnings.catch_warnings():  # all-NaN columns (offsets never reached) are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        post_stats = np.concatenate([np.nanpercentile(post, Q, axis=0).T, np.nanmean(post, 0)[:, None]], 1)
    post_count = np.isfinite(post).sum(0)
    per = dict(t_esc=t_esc, t_handover=t_h, in_region_at_escape=inD_at_esc, region_exits=exits,
               last_region_exit=last_exit, saddle_reentries=reentry, basin_switches=switches, t_diverged=t_div,
               tail_mean_loss=np.where(alive, tail / max(tail_n, 1), np.nan), sign_at_handover=sign_h)
    return dict(grid=G, quantiles=np.array(Q), post_grid=G, post_stats=post_stats, post_count=post_count,
                keep_a=keep_a, keep_b=keep_b, **{f"stat_{k}": v for k, v in stats.items()}, **per)


def summarize(sim, th, step):
    t_e, t_h = sim["t_esc"], sim["t_handover"]
    S = len(t_e)
    esc, ho = t_e >= 0, t_h >= 0
    div = sim["t_diverged"] >= 0
    immediate = esc & sim["in_region_at_escape"]
    eventual = ho & ~immediate
    retained = ho & (sim["region_exits"] == 0) & ~div
    dtau = (t_h - t_e)[esc & ho]
    f = lambda m: round(100.0 * m.sum() / S, 2)
    q = lambda v: None if len(v) == 0 else [float(x) for x in np.percentile(v, [5, 25, 50, 75, 95])]
    return dict(n_trajectories=S, pct_escaped=f(esc), pct_immediate_handover=f(immediate),
                pct_eventual_handover=f(eventual), pct_failed_handover=f(~ho), pct_diverged=f(div),
                pct_retained_no_region_exit=f(retained), pct_any_region_exit=f(ho & (sim["region_exits"] > 0)),
                pct_saddle_reentry=f(sim["saddle_reentries"] > 0), pct_basin_switch_after_handover=f(sim["basin_switches"] > 0),
                t_escape_mean=(float(t_e[esc].mean()) if esc.any() else None), t_escape_pct=q(t_e[esc]),
                t_handover_pct=q(t_h[ho]), delta_tau_pct=q(dtau), delta_tau_min_max=(None if len(dtau) == 0 else [int(dtau.min()), int(dtau.max())]),
                region_exits_pct=q(sim["region_exits"][ho]), last_exit_pct=q(sim["last_region_exit"][sim["last_region_exit"] >= 0]),
                tail_mean_loss_pct=q(sim["tail_mean_loss"][np.isfinite(sim["tail_mean_loss"])]),
                escape_time_vs_thm3_bound=(None if not esc.any() else float(t_e[esc].mean() / th["thm3_tau_bound"])))


def main():
    os.makedirs(RAW, exist_ok=True)
    out = {}
    for q in (2, 4):
        th = theory(q)
        th["torch_network_gradient_check"] = torch_check(q)
        out[f"q{q}"] = dict(theory=th, runs={})
        print(f"q={q} theory:", json.dumps({k: th[k] for k in ("mu", "a_crit", "b_crit", "r", "r_lin", "rho1", "rho2",
                                                             "rho3", "thm3_tau_bound", "gd_escape_iters_from_near_init")}))
        for ii, (iname, init) in enumerate(INITS.items()):
            cfgs = [(o, s, False) for o in ("sgd", "fim", "iso") for s in ("const", "rm")] + [("fim", "const", True)]
            for opt, step, off in cfgs:
                name = f"q{q}_{iname}_{opt}_{step}" + ("_noiseoff" if off else "")
                sim = simulate(q, th, init, {"sgd": "none"}.get(opt, opt), step, off, seed=100 * q + ii)
                np.savez_compressed(os.path.join(RAW, name + ".npz"), **sim)
                out[f"q{q}"]["runs"][name] = summarize(sim, th, step)
                s = out[f"q{q}"]["runs"][name]
                print(f"  {name:32s} esc {s['pct_escaped']:6.1f}%  immediate {s['pct_immediate_handover']:6.1f}%  "
                      f"exit {s['pct_any_region_exit']:6.1f}%  switch {s['pct_basin_switch_after_handover']:6.1f}%  "
                      f"div {s['pct_diverged']:5.1f}%  tail {s['tail_mean_loss_pct'] and round(s['tail_mean_loss_pct'][2], 6)}", flush=True)
    json.dump(out, open(os.path.join(HERE, "summary.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
