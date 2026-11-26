"""Generalisation of ../run_experiment2.py to saddle degree p, step size eta, escape radius r = r_frac * r_lin and
noise multiplier noise_mult. With p=4, eta=0.01, r_frac=0.5, noise_mult=1 it must reproduce ../summary.json exactly
(see check_regression.py).

Model: tied p-layer deep linear net f_{a,b}(x) = (a^p x, b x), x in {-1,+1}, targets (x, 0);
       L_{p,q}(a,b) = 0.1[(a^p - 1)^q + b^2], L* = 0, minima (+-1, 0). Every sample has the same loss (x^2 = x^4 = 1),
       so the mini-batch noise M_{n+1} is identically 0 and SGD is plain gradient descent.
p = 2 is a strict saddle (paper Sec. 3: Lemma 1, Thm 1, Thm 2); p >= 3 a p-th order degenerate saddle (Sec. 4).
"""
import functools, math
import numpy as np
import sympy as sp

CPRIME, KAPPA, N_TRAJ, N_KEEP = 0.5, 0.75, 200, 20
INITS = {"saddle": (0.0, 0.0), "near": (0.01, 0.01)}
DESCENT_LOSS = 0.09   # "descent" event: L < 0.09 (saddle loss is 0.1)


# ----------------------------------------------------------------------------- analytic model
def loss(p, q, a, b):
    return 0.1 * ((a ** p - 1) ** q + b ** 2)


def grad(p, q, a, b):
    return 0.1 * q * (a ** p - 1) ** (q - 1) * p * a ** (p - 1), 0.2 * b


def ratio(p, q, a, b):
    """Measured R_PL = |grad|^2 / (2(L-L*)) for q=2, R_KL = |grad| / (L-L*)^kappa for q=4 (L* = 0)."""
    ga, gb = grad(p, q, a, b)
    g2, L = ga ** 2 + gb ** 2, loss(p, q, a, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        return g2 / (2 * L) if q == 2 else np.sqrt(g2) / L ** KAPPA


def ratio_lower_bound(p, q, a, b):
    """Pointwise lower bound psi <= ratio.  A = (a^p-1)^q, B = b^2.
    q=2: R_PL = (0.2 p^2 a^{2p-2} A + 0.2 B)/(A+B) >= 0.2 min(p^2 a^{2p-2}, 1)   (weighted average).
    q=4: R_KL^2 = (0.16 p^2 a^{2p-2} A^1.5 + 0.04 B)/(0.1^1.5 (A+B)^1.5); (A+B)^1.5 <= sqrt2 (A^1.5 + B^1.5) and
         0.04 B >= (0.04/|b|) B^1.5  =>  R_KL >= sqrt(min(0.16 p^2 a^{2p-2}, 0.04/|b|) / (0.1^1.5 sqrt2))."""
    c = 0.2 * p ** 2 if q == 2 else 0.16 * p ** 2
    if q == 2:
        return np.minimum(c * a ** (2 * p - 2), 0.2)
    with np.errstate(divide="ignore"):
        m = np.minimum(c * a ** (2 * p - 2), 0.04 / np.abs(b))
    return np.sqrt(m / (0.1 ** 1.5 * math.sqrt(2)))


# ----------------------------------------------------------------------------- theory
@functools.lru_cache(None)
def symbolic(p, q):
    """Exact checks with sympy: order of the saddle along u = e_a, vanishing mixed p-th derivatives, stationary points."""
    A, B, T = sp.symbols("a b t", real=True)
    Ls = sp.Rational(1, 10) * ((A ** p - 1) ** q + B ** 2)
    du = [sp.diff(Ls.subs({A: T, B: 0}), T, k).subs(T, 0) for k in range(1, p + 1)]
    # terms entering <grad^p L (Z u + v)^{p-1}, u>: at least one a and one b. (k = 0, pure b, is the b-curvature: 0.2 at p = 2)
    mixed = [sp.diff(Ls, A, k, B, p - k).subs({A: 0, B: 0}) for k in range(1, p)]
    stat = sp.solve([sp.diff(Ls, A), sp.diff(Ls, B)], [A, B], dict=True)
    hess0 = sp.hessian(Ls, (A, B)).subs({A: 0, B: 0})
    hess1 = sp.hessian(Ls, (A, B)).subs({A: 1, B: 0})
    assert du[:p - 1] == [0] * (p - 1) and du[p - 1] < 0, f"saddle is not of order p={p} along u=e_a"
    assert all(m == 0 for m in mixed), "mixed p-th derivatives must vanish (L_p = 0)"
    assert {(s[A], s[B]) for s in stat} == {(0, 0), (1, 0), (-1, 0)}, stat
    assert math.isclose(float(du[p - 1]), -0.1 * q * math.factorial(p), rel_tol=1e-12)
    return float(du[p - 1]), [float(x) for x in hess0.diagonal()], [float(x) for x in hess1.diagonal()], \
        str(hess0.tolist()), str(hess1.tolist()), str([(s[A], s[B]) for s in stat])


@functools.lru_cache(None)
def gd_escape(p, q, eta, r, a0=INITS["near"][0], n_max=2_000_000):
    """Deterministic GD (= SGD here) escape time from a = a0 (b does not affect a). Exact up to n_max iterations,
    otherwise the leading-order ODE estimate da/dn = eta*lead*a^{p-1} (labelled as an estimate)."""
    x, n = a0, 0
    while abs(x) < r and n < n_max:
        x -= eta * grad(p, q, x, 0.0)[0]; n += 1
    if abs(x) >= r:
        return n, "exact"
    lead = 0.1 * q * math.factorial(p) / math.factorial(p - 1)
    est = math.log(r / a0) / (eta * lead) if p == 2 else (a0 ** (2 - p) - r ** (2 - p)) / ((p - 2) * eta * lead)
    return est, "ODE estimate"


def theory(p, q, eta=0.01, r_frac=0.5, noise_mult=1.0, rho_radius_frac=0.5):
    mu_deg, h0, h1, hess0, hess1, stat = symbolic(p, q)
    lead = abs(mu_deg) / math.factorial(p - 1)        # grad_a = -lead a^{p-1} + R_p(a)
    ag = np.linspace(1e-6, 1.0, 400_001)
    Rp = np.abs(grad(p, q, ag, 0.0)[0] + lead * ag ** (p - 1))
    bad = Rp > 0.5 * lead * ag ** (p - 1)             # r_lin: leading term dominates the remainder by 2x
    r_lin = float(ag[np.argmax(bad)] if bad.any() else ag[-1])
    m = ag <= r_lin
    Kp_prime = float(np.max(Rp[m] / ag[m] ** p))      # |R_p| <= K'_p |Z|^p on |Z| <= r_lin
    Kp = lead + Kp_prime * r_lin                      # Eq. (31) with L_p = 0, C_0 = 0
    Kp1 = float(np.max(np.abs(loss(p, q, ag[m], 0) - 0.1 - mu_deg / math.factorial(p) * ag[m] ** p) / ag[m] ** p))
    opnorm = abs(mu_deg) / math.factorial(p)          # ||grad^p L|| / p!  (only the a^p component)
    dL = 0.1
    r = r_frac * r_lin
    r_rho = rho_radius_frac * r_lin                   # radius at which the noise is set (fixed across the r sweep)

    a_c4 = (lead - Kp_prime * r) ** 2                 # Thm 4 constant 'a' with R(r0) = 1
    b_c4 = r ** p * (opnorm + Kp1)                    # Thm 4 constant 'b'
    out = {}
    if q == 2:
        if p == 2:   # Thm 2, Eq. (17): r^2 mu_min^2 / (dL + r^2 |mu_min|/2) >= 2 mu  (already the R_PL convention)
            mu = r ** 2 * mu_deg ** 2 / (2 * (dL + r ** 2 * abs(mu_deg) / 2))
            mu_T4 = 2 * mu
        else:        # Thm 4 uses |grad|^2 >= mu_T4 (L - L*); R_PL uses 2 mu, so mu = mu_T4 / 2
            mu_T4 = r ** (2 * (p - 1)) * a_c4 / (dL + b_c4)
            mu = mu_T4 / 2
        mu_cert = min(mu, 0.2)                        # pitfall 1: along b the PL ratio is exactly 0.2
        c2 = 0.2 * p ** 2
        a_crit = (mu_cert / c2) ** (1 / (2 * p - 2))
        b_crit = math.inf
        exact_r_b0 = (mu / c2) ** (1 / (2 * p - 2))   # PL ratio on b = 0 is c2 a^{2p-2}
        rhs35 = 2 * Kp / CPRIME * (mu_T4 * (dL + b_c4) / a_c4) ** (p / (2 * (p - 1)))       # Eq. (35)
    else:            # Cor. 1 (p = 2: the strict-saddle note at the end of Sec. 4.1 is the same formula)
        mu = r ** (p - 1) * math.sqrt(a_c4) / (dL + b_c4) ** KAPPA
        mu_cert = mu
        a_crit = (mu ** 2 * 0.1 ** 1.5 * math.sqrt(2) / (0.16 * p ** 2)) ** (1 / (2 * p - 2))
        b_crit = 0.04 / (0.16 * p ** 2 * a_crit ** (2 * p - 2))
        exact_r_b0 = (mu * 0.1 ** KAPPA / (0.4 * p)) ** (1 / (p - 1))
        rhs35 = 2 * Kp / CPRIME * mu ** (p / (p - 1)) * (dL + b_c4) ** (KAPPA * p / (p - 1)) / a_c4 ** (p / (2 * (p - 1)))  # Eq. (41)
    inf_bound = float(ratio_lower_bound(p, q, np.array(a_crit), np.array(0.0 if q == 2 else b_crit)))
    ga_ = np.concatenate([-np.geomspace(a_crit, 1.6, 801), np.geomspace(a_crit, 1.6, 801)])
    bmax = 10.0 if q == 2 else b_crit
    gb_ = np.concatenate([-np.geomspace(1e-8, bmax, 600), [0.0], np.geomspace(1e-8, bmax, 600)])
    AA, BB = np.meshgrid(ga_, gb_)
    with np.errstate(over="ignore", invalid="ignore"):
        Lg, Rg = loss(p, q, AA, BB), ratio(p, q, AA, BB)
    keep = (Lg > 1e-14) & np.isfinite(Rg)              # exclude the minimum itself (0/0) and float overflow
    grid_inf = float(np.min(Rg[keep]))
    # mathematical self-checks (not outcomes): the bound is attained at the boundary and is below the exact ratio
    assert inf_bound >= mu_cert * (1 - 1e-9) and grid_inf >= inf_bound * (1 - 1e-9), (p, q, mu_cert, inf_bound, grid_inf)

    rho3 = float(np.max(grad(p, q, np.linspace(0, r_rho, 100_001), 0.0)[0] ** 2))  # g_a^2 bound on |a| <= r_rho
    rho2 = rho3
    rho1 = noise_mult * (rho2 + rho3) * math.sqrt(2 * Kp * r_rho ** p / (eta * CPRIME))
    r0 = (eta * CPRIME * rho1 ** 2 / (2 * Kp * (rho2 + rho3) ** 2)) ** (1 / p)
    assert abs(r0 - noise_mult ** (2 / p) * r_rho) < 1e-12, "Eq. (31) algebra"
    rho3_esc = float(np.max(grad(p, q, np.linspace(0, r, 100_001), 0.0)[0] ** 2))  # g_a^2 bound on the escape region
    lhs35 = eta * (rho1 / (rho2 + rho3_esc)) ** 2

    aa = np.linspace(-1.2, 1.2, 240_001)              # smoothness: eta <= 1/ell on |a| <= 1.2
    ell = float(max(np.max(np.abs(np.gradient(grad(p, q, aa, 0.0)[0], aa[1] - aa[0]))), 0.2))
    hess_min_a = h1[0]
    step_max = eta * (math.sqrt(rho3_esc) + rho1 / rho2)  # largest |Delta a| per step inside the r-slab
    K_over = (r + step_max) ** 2 - r ** 2
    c_fac = 1.0 if p == 2 else 1 / (1 - CPRIME)       # Thm 1 (strict) has no 1/(1-c') factor; Thm 3 does
    tau_bound = c_fac * ((rho2 + rho3_esc) / (eta * rho1)) ** 2 * (r ** 2 + K_over)
    n0_rm = max(1000, math.ceil(tau_bound))
    gd_n, gd_kind = gd_escape(p, q, eta, r)

    if p == 2:   # strict-saddle extras (Sec. 3)
        pos = [x for x in h0 if x > 0]
        mu_max, mu_min_pos = max(pos), min(pos)
        kappa_cone = 1 / (1 + mu_max / mu_min_pos)
        R2 = np.abs(grad(p, q, ag[m], 0.0)[0] - mu_deg * ag[m])     # |grad - H theta| along u
        L2 = float(np.max(R2 / ag[m] ** 2))
        eq8_radius = kappa_cone * abs(mu_deg) / (2 * L2)
        out.update(kappa_cone=kappa_cone, L2=L2, eq8_radius=eq8_radius,
                   eq8_r_over_radius=r / eq8_radius,   # = 1 at r = r_lin/2 (equality); r_lin grid step is 2.5e-6
                   flag_eq8_radius=bool(r <= eq8_radius * (1 + 1e-5)),
                   thm2_gap_bound=kappa_cone ** 2 * abs(mu_deg) ** 3 / (2 * mu * L2 ** 2),
                   flag_thm2_gap=bool(dL <= kappa_cone ** 2 * abs(mu_deg) ** 3 / (2 * mu * L2 ** 2)))
    out.update(
        p=p, q=q, eta=eta, r_frac=r_frac, noise_mult=noise_mult, rho_radius_frac=rho_radius_frac,
        saddle_type="strict" if p == 2 else "degenerate", mu_deg=mu_deg, lead=lead, hessian_at_saddle=hess0,
        hessian_at_min=hess1, stationary_points=stat, r_lin=r_lin, Kp_prime=Kp_prime, Kp=Kp, Kp1=Kp1,
        opnorm_over_pfact=opnorm, dL=dL, r=r, r_rho=r_rho, thm4_a=a_c4, thm4_b=b_c4, mu=mu, mu_certified=mu_cert,
        mu_source=("Thm 2 Eq.(17)" if (q == 2 and p == 2) else "Thm 4 (mu_T4/2)" if q == 2 else "Cor. 1"),
        mu_convention=("R_PL=|g|^2/(2(L-L*))" if q == 2 else "R_KL=|g|/(L-L*)^0.75"),
        a_crit=a_crit, b_crit=b_crit, exact_radius_along_b0=exact_r_b0, certified_inf_bound=inf_bound,
        grid_inf_ratio_in_D=grid_inf, rho1=rho1, rho2=rho2, rho3=rho3, rho3_escape_region=rho3_esc,
        r0_eq31=r0, eq35_lhs=lhs35, eq35_rhs=rhs35, c_prime=CPRIME, smoothness_ell=ell,
        hessian_min_a=hess_min_a, noise_step_at_zero_grad=eta * rho1 / rho2,
        random_walk_escape_estimate=(r / (eta * rho1 / rho2)) ** 2, thm3_overshoot_K=K_over,
        escape_bound_theorem=("Thm 1" if p == 2 else "Thm 3"), thm3_tau_bound=tau_bound, n0_rm=n0_rm,
        iso_sigma=rho1 / rho2, gd_escape_iters_from_near_init=gd_n, gd_escape_kind=gd_kind,
        thm4_a_positive=bool(lead - Kp_prime * r > 0),
        flag_lemma2_radius=bool(r <= min(r_lin, r0) * (1 + 1e-12)),
        flag_rho3_bounds_escape_region=bool(rho3 >= rho3_esc * (1 - 1e-12)),
        flag_r_ge_acrit=bool(r >= a_crit), flag_noise_threshold=bool(lhs35 >= rhs35 * (1 - 1e-9)),
        flag_eta_le_1_over_ell=bool(eta <= 1 / ell),
        flag_eta_stable_at_min=bool(eta < 2 / max(hess_min_a, 0.2)),
        flag_n0_ge_bound=bool(n0_rm >= tau_bound), flag_mu_attainable=bool(mu <= 0.2) if q == 2 else True)
    return out


def torch_check(p, q):
    """The actual tied p-layer linear network on the 4096-point dataset: autograd mini-batch gradients equal the
    analytic gradient, and every mini-batch gives the same gradient (so M_{n+1} == 0)."""
    import torch
    g = torch.Generator().manual_seed(0)
    X = (torch.randint(0, 2, (4096,), generator=g) * 2 - 1).double()
    Y = torch.stack([X, torch.zeros_like(X)], 1)
    for a0, b0 in [(0.3, -0.7), (1.1, 0.2), (-0.6, 1.5)]:
        grads = []
        for idx in torch.randperm(4096, generator=g)[:384].view(3, 128):
            a = torch.tensor(a0, dtype=torch.float64, requires_grad=True)
            b = torch.tensor(b0, dtype=torch.float64, requires_grad=True)
            h = X[idx]
            for _ in range(p):                        # p tied weight layers, identity activation
                h = a * h
            out = torch.stack([h, b * X[idx]], 1)
            (0.1 * ((out[:, 0] - Y[idx, 0]) ** q + (out[:, 1] - Y[idx, 1]) ** 2)).mean().backward()
            grads.append([a.grad.item(), b.grad.item()])
        ga, gb = grad(p, q, a0, b0)
        assert np.allclose(grads, [[ga, gb]] * 3, rtol=1e-12, atol=1e-12), (p, q, grads, ga, gb)
    return True


# ----------------------------------------------------------------------------- simulation
def grid(N):
    return np.unique(np.concatenate([np.arange(0, 500), np.arange(500, 5000, 10), np.arange(5000, N + 1, 100), [N]]))


def _new_events(S):
    return dict(t_esc=np.full(S, -1), t_handover=np.full(S, -1), in_region_at_escape=np.zeros(S, bool),
                sign_at_handover=np.zeros(S), region_exits=np.zeros(S, int), last_region_exit=np.full(S, -1),
                saddle_reentries=np.zeros(S, int), last_reentry=np.full(S, -1), basin_switches=np.zeros(S, int),
                last_switch=np.full(S, -1), in_region_at_end=np.zeros(S, bool),
                _prev_inD=np.zeros(S, bool), _prev_sticky=np.ones(S, bool))


def simulate(p, q, th, init, noise, step="const", schedule="const", seed=0, N=50_000, specs=None, full=True):
    """noise: 'none' (SGD = GD), 'fim' (Eq. 1), 'iso' (Gaussian, sigma = rho1/rho2), 'sign' (+-rho1/rho2, i.e. Eq. 1
    without the gradient-dependent scaling). step: 'const' or 'rm' (alpha_n = eta n0/(n0+n), Thm 6).
    schedule (after the first certified handover of specs[0]): 'const' (paper), 'off' (noise disabled, diagnostic),
    'decay' (noise x n0/(n0 + n - tau_h), diagnostic).  specs: list of event definitions (r, a_crit, b_crit) evaluated
    on the SAME trajectories (radius sweep); specs[0] drives the schedule and the post-handover statistics."""
    S = N_TRAJ
    eta, rho1, rho2, n0 = th["eta"], th["rho1"], th["rho2"], th["n0_rm"]
    specs = specs or [th]
    rng = np.random.default_rng(seed)
    a, b = np.full(S, init[0]), np.full(S, init[1])
    sig = rho1 / rho2
    G = grid(N)
    gi = np.full(N + 1, -1)
    gi[G] = np.arange(len(G))
    Q = [5, 25, 50, 75, 95]
    keys = ("loss", "ratio", "ratio_lb", "dist_saddle", "dist_min", "noise_step", "abs_a", "abs_b") if full \
        else ("loss", "abs_a", "abs_b")
    stats = {k: np.full((len(G), len(Q) + 1), np.nan) for k in keys}
    keep_a = np.zeros((len(G), N_KEEP), np.float32)
    keep_b = np.zeros((len(G), N_KEEP), np.float32)
    post = np.full((S, len(G)), np.nan, np.float32) if full else None
    evs = [_new_events(S) for _ in specs]
    t_div = np.full(S, -1); t_desc = np.full(S, -1)
    alive = np.ones(S, bool)
    prev_sign = np.sign(a)
    tail = np.zeros(S); tail_n = 0
    for n in range(N + 1):
        absa = np.abs(a)
        sg = np.sign(a)
        for sp_, ev in zip(specs, evs):
            inD = (absa >= sp_["a_crit"]) & (np.abs(b) <= sp_["b_crit"]) & alive
            sticky = (absa < sp_["r"]) & alive
            new_e = (ev["t_esc"] < 0) & ~sticky & alive
            ev["t_esc"][new_e] = n; ev["in_region_at_escape"][new_e] = inD[new_e]
            new_h = (ev["t_handover"] < 0) & inD
            ev["t_handover"][new_h] = n; ev["sign_at_handover"][new_h] = sg[new_h]
            after_h = (ev["t_handover"] >= 0) & (ev["t_handover"] < n) & alive
            ex = after_h & ev["_prev_inD"] & ~inD
            ev["region_exits"] += ex; ev["last_region_exit"][ex] = n
            re = (ev["t_esc"] >= 0) & (ev["t_esc"] < n) & alive & ~ev["_prev_sticky"] & sticky
            ev["saddle_reentries"] += re; ev["last_reentry"][re] = n
            sw = after_h & (sg != prev_sign) & (sg != 0) & (prev_sign != 0)
            ev["basin_switches"] += sw; ev["last_switch"][sw] = n
            ev["_prev_inD"], ev["_prev_sticky"] = inD, sticky
        prev_sign = np.where(sg != 0, sg, prev_sign)
        t_h = evs[0]["t_handover"]
        L = loss(p, q, a, b)
        t_desc[(t_desc < 0) & alive & (L < DESCENT_LOSS)] = n
        if full:
            k = n - t_h
            ok = (t_h >= 0) & alive
            kk = gi[np.clip(k, 0, N)]
            ok &= kk >= 0
            post[ok, kk[ok]] = L[ok]
        if n >= N - 10_000:
            tail += L; tail_n += 1
        ga, gb = grad(p, q, a, b)
        if noise == "fim":
            s = rng.integers(0, 2, (2, S)) * 2 - 1
            ua, ub = s[0] * rho1 / (rho2 + ga ** 2), s[1] * rho1 / (rho2 + gb ** 2)
        elif noise == "iso":
            ua, ub = rng.normal(0.0, sig, (2, S))
        elif noise == "sign":
            s = rng.integers(0, 2, (2, S)) * 2 - 1
            ua, ub = s[0] * sig, s[1] * sig
        else:
            ua, ub = np.zeros(S), np.zeros(S)
        if schedule == "off":
            off = t_h >= 0
            ua, ub = np.where(off, 0.0, ua), np.where(off, 0.0, ub)
        elif schedule == "decay":
            fac = np.where(t_h >= 0, n0 / (n0 + n - np.maximum(t_h, 0)), 1.0)
            ua, ub = ua * fac, ub * fac
        al = eta if step == "const" else eta * n0 / (n0 + n)
        if gi[n] >= 0:
            j = gi[n]
            vals = dict(loss=L, abs_a=absa, abs_b=np.abs(b))
            if full:
                sign_h = evs[0]["sign_at_handover"]
                dest = np.where(t_h >= 0, sign_h, 1.0)
                dmin = np.where(t_h >= 0, np.hypot(a - dest, b), np.hypot(absa - 1, b))
                vals.update(ratio=ratio(p, q, a, b), ratio_lb=ratio_lower_bound(p, q, a, b), dist_saddle=np.hypot(a, b),
                            dist_min=dmin, noise_step=al * np.hypot(ua, ub))
                keep_a[j], keep_b[j] = a[:N_KEEP], b[:N_KEEP]
            for key in keys:
                v = np.where(alive, vals[key], np.nan)
                if np.isfinite(v).any():
                    stats[key][j, :len(Q)] = np.nanpercentile(v, Q)
                    stats[key][j, -1] = np.nanmean(v)
        if n == N:
            break
        with np.errstate(over="ignore", invalid="ignore"):
            a = a - al * (ga + ua)
            b = b - al * (gb + ub)
        div = alive & (~np.isfinite(a) | ~np.isfinite(b) | (np.abs(a) > 10) | (np.abs(b) > 1e3))
        t_div[div] = n + 1
        alive &= ~div
        a = np.where(alive, a, np.nan); b = np.where(alive, b, np.nan)
    for ev in evs:
        ev["in_region_at_end"] = ev.pop("_prev_inD") & alive
        ev.pop("_prev_sticky")
    common = dict(grid=G, quantiles=np.array(Q), t_diverged=t_div, t_descent=t_desc,
                  tail_mean_loss=np.where(alive, tail / max(tail_n, 1), np.nan),
                  **{f"stat_{k}": v for k, v in stats.items()})
    if full:
        import warnings
        with warnings.catch_warnings():  # all-NaN columns (offsets never reached) are expected
            warnings.simplefilter("ignore", RuntimeWarning)
            common["post_stats"] = np.concatenate([np.nanpercentile(post, Q, axis=0).T, np.nanmean(post, 0)[:, None]], 1)
        common.update(post_grid=G, post_count=np.isfinite(post).sum(0), keep_a=keep_a, keep_b=keep_b)
    return common, evs


# ----------------------------------------------------------------------------- statistics
def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [100 * max(0.0, c - h), 100 * min(1.0, c + h)]


def summarize(common, ev, th, N, seed=0):
    """Original Experiment-2 fields (identical code, for the regression test) plus censoring-aware statistics."""
    t_e, t_h = ev["t_esc"], ev["t_handover"]
    S = len(t_e)
    esc, ho = t_e >= 0, t_h >= 0
    div = common["t_diverged"] >= 0
    immediate = esc & ev["in_region_at_escape"]
    eventual = ho & ~immediate
    retained = ho & (ev["region_exits"] == 0) & ~div
    dtau = (t_h - t_e)[esc & ho]
    f = lambda m: round(100.0 * m.sum() / S, 2)
    q = lambda v: None if len(v) == 0 else [float(x) for x in np.percentile(v, [5, 25, 50, 75, 95])]
    out = dict(n_trajectories=S, pct_escaped=f(esc), pct_immediate_handover=f(immediate),
               pct_eventual_handover=f(eventual), pct_failed_handover=f(~ho), pct_diverged=f(div),
               pct_retained_no_region_exit=f(retained), pct_any_region_exit=f(ho & (ev["region_exits"] > 0)),
               pct_saddle_reentry=f(ev["saddle_reentries"] > 0), pct_basin_switch_after_handover=f(ev["basin_switches"] > 0),
               t_escape_mean=(float(t_e[esc].mean()) if esc.any() else None), t_escape_pct=q(t_e[esc]),
               t_handover_pct=q(t_h[ho]), delta_tau_pct=q(dtau), delta_tau_min_max=(None if len(dtau) == 0 else [int(dtau.min()), int(dtau.max())]),
               region_exits_pct=q(ev["region_exits"][ho]), last_exit_pct=q(ev["last_region_exit"][ev["last_region_exit"] >= 0]),
               tail_mean_loss_pct=q(common["tail_mean_loss"][np.isfinite(common["tail_mean_loss"])]),
               escape_time_vs_thm3_bound=(None if not esc.any() else float(t_e[esc].mean() / th["thm3_tau_bound"])))
    # censoring-aware escape statistics: min(tau, N) is a lower bound for tau; its mean lower-bounds E[tau]
    cens = np.where(esc, t_e, N).astype(float)
    boot = np.random.default_rng(seed + 12345).choice(cens, (2000, S)).mean(1)
    desc = common["t_descent"] >= 0
    tl = common["tail_mean_loss"][np.isfinite(common["tail_mean_loss"])]
    out.update(
        n_iter=N, n_escaped=int(esc.sum()), n_censored=int((~esc).sum()), pct_escaped_wilson95=wilson(int(esc.sum()), S),
        t_escape_mean_escaped_only=out["t_escape_mean"], t_escape_censored_mean_lower_bound=float(cens.mean()),
        t_escape_censored_mean_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        censored_mean_vs_bound=float(cens.mean() / th["thm3_tau_bound"]),
        pct_descent=f(desc), pct_descent_wilson95=wilson(int(desc.sum()), S), t_descent_pct=q(common["t_descent"][desc]),
        pct_handover=f(ho), pct_handover_at_or_before_escape=f(esc & ho & (t_h <= t_e)),
        pct_in_region_at_escape_wilson95=wilson(int(immediate.sum()), S),
        pct_saddle_reentry_wilson95=wilson(int((ev["saddle_reentries"] > 0).sum()), S),
        last_reentry_max=int(ev["last_reentry"].max()), last_reentry_pct=q(ev["last_reentry"][ev["last_reentry"] >= 0]),
        pct_basin_switch_wilson95=wilson(int((ev["basin_switches"] > 0).sum()), S),
        last_switch_max=int(ev["last_switch"].max()), last_region_exit_max=int(ev["last_region_exit"].max()),
        pct_in_region_at_end=f(ev["in_region_at_end"]), pct_in_region_at_end_wilson95=wilson(int(ev["in_region_at_end"].sum()), S),
        floor_median_tail_loss=(float(np.median(tl)) if len(tl) else None))
    return out
