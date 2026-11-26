"""Tables, convergence rates, theorem checks and figures A1-A6 from results/*.json and raw/*.npz.
Writes results/table_all_configs.csv, results/convergence_rates.json, results/theorem_checks.json, figures/*.{png,pdf}."""
import csv, glob, json, math, os
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ablation_core import loss, theory

HERE = os.path.dirname(os.path.abspath(__file__))
RES, RAW, FIG = (os.path.join(HERE, d) for d in ("results", "raw", "figures"))
os.makedirs(FIG, exist_ok=True)
PS, QS, ETAS, MULTS, RFS = (2, 4, 6, 8), (2, 4), (0.0025, 0.005, 0.01, 0.02), (0.25, 0.5, 1.0, 2.0, 4.0), (0.25, 0.5, 0.75, 1.0)
CO = dict(none="#2a78d6", fim="#eb6834", iso="#1baf7a", sign="#6250d6")
LAB = dict(none="SGD (= GD)", fim="Eq. (1)", iso="isotropic", sign="const.-magnitude sign")
CP = dict(zip(PS, plt.cm.viridis([0.05, 0.38, 0.65, 0.92])))
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.5, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})

R = {}
for f in glob.glob(os.path.join(RES, "S*.json")):
    r = json.load(open(f)); R[r["name"]] = r
print(len(R), "configurations")
S, T, C = (lambda n: R[n]["summary"]), (lambda n: R[n]["theory"]), (lambda n: R[n]["config"])
raw = lambda n: np.load(os.path.join(RAW, n + ".npz"))
FLAGS = [k for k in next(iter(R.values()))["theory"] if k.startswith("flag_")] + ["flag_eq8_radius", "flag_thm2_gap"]


def name_base(p, q, noise, sweep="S1", eta=0.01, mult=1.0, rf=0.5, var="thm"):
    """Name of the configuration at a sweep point; the base point (eta=0.01, mult=1, r=r_lin/2) lives in S1."""
    if eta == 0.01 and mult == 1.0 and rf == 0.5:
        return f"S1_p{p}_q{q}_saddle_{'sgd' if noise == 'none' else noise}_const"
    if eta != 0.01:
        return f"S2_p{p}_q{q}_eta{eta}_{var}_{noise}"
    if mult != 1.0:
        return f"S3_p{p}_q{q}_mult{mult}_{noise}"
    return f"S4_p{p}_q{q}_{noise}_r{rf}"


def save(fig, n):
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{n}.{e}"), dpi=300)
    plt.close(fig)


def heat(ax, M, rows, cols, title, fmt="{:.0f}", vmax=100, cmap="Oranges"):
    ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, "–" if np.isnan(v) else fmt.format(v), ha="center", va="center", fontsize=6.5,
                    color="white" if (not np.isnan(v) and v > 0.6 * vmax) else "black")
    ax.set_xticks(range(len(cols)), cols); ax.set_yticks(range(len(rows)), rows); ax.set_title(title); ax.grid(False)


def escape_assumptions(th):
    """Assumptions of the escape bound: Thm 1 (p=2): Eq. (8) radius + rho3 bound; Thm 3 (p>=3): Lemma 2 radius + rho3."""
    return bool((th.get("flag_eq8_radius") if th["p"] == 2 else th["flag_lemma2_radius"]) and th["flag_rho3_bounds_escape_region"])


def bound_verdict(s, th):
    lo, hi = s["t_escape_censored_mean_ci95"]
    if s["n_censored"] == 0:
        return "below bound" if hi <= th["thm3_tau_bound"] else ("above bound" if lo > th["thm3_tau_bound"] else "CI contains bound")
    return "censored lower bound exceeds bound" if s["t_escape_censored_mean_lower_bound"] > th["thm3_tau_bound"] \
        else "inconclusive (censored)"


def cochran_armitage(x, n, s):
    x, n, s = map(np.asarray, (x, n, s))
    N, pbar = n.sum(), x.sum() / n.sum()
    var = pbar * (1 - pbar) * ((n * s ** 2).sum() - (n * s).sum() ** 2 / N)
    if var <= 0:
        return dict(z=None, p_value=None, note="all proportions 0% or 100%")
    z = float((s * (x - n * pbar)).sum() / math.sqrt(var))
    return dict(z=z, p_value=float(2 * norm.sf(abs(z))))


# ============================================================================ table
cols_c = ["sweep", "p", "q", "init", "noise", "step", "sched", "eta", "mult", "r_frac", "N", "variant", "raw_file", "event_prefix"]
cols_t = ["saddle_type", "r_lin", "r", "mu", "mu_certified", "mu_source", "a_crit", "b_crit", "exact_radius_along_b0", "rho1",
          "rho2", "rho3", "rho3_escape_region", "r0_eq31", "eq35_lhs", "eq35_rhs", "smoothness_ell", "noise_step_at_zero_grad",
          "random_walk_escape_estimate", "escape_bound_theorem", "thm3_tau_bound", "n0_rm", "gd_escape_iters_from_near_init",
          "gd_escape_kind"] + FLAGS
cols_s = ["pct_escaped", "pct_escaped_wilson95", "n_censored", "t_escape_mean_escaped_only", "t_escape_censored_mean_lower_bound",
          "t_escape_censored_mean_ci95", "censored_mean_vs_bound", "pct_descent", "pct_handover", "pct_in_region_at_escape_wilson95",
          "pct_immediate_handover", "pct_handover_at_or_before_escape", "pct_any_region_exit", "pct_saddle_reentry",
          "pct_saddle_reentry_wilson95", "last_reentry_max", "pct_basin_switch_after_handover", "pct_basin_switch_wilson95",
          "last_switch_max", "last_region_exit_max", "pct_in_region_at_end", "pct_in_region_at_end_wilson95", "pct_diverged",
          "floor_median_tail_loss"]
with open(os.path.join(RES, "table_all_configs.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["name"] + cols_c + cols_t + cols_s)
    for n in sorted(R):
        w.writerow([n] + [C(n).get(k) for k in cols_c] + [T(n).get(k) for k in cols_t] +
                   [json.dumps(S(n).get(k)) if isinstance(S(n).get(k), list) else S(n).get(k) for k in cols_s])

# ============================================================================ convergence rates
rates = {}
for p in PS:
    for q in QS:
        n = f"S1_p{p}_q{q}_saddle_fim_off"
        d = raw(n)
        k, med = d["post_grid"].astype(float), d["post_stats"][:, 2]
        th = T(n)
        if q == 2:
            m = (med > 1e-25) & (med < 1e-5)
            sl = float(np.polyfit(k[m], np.log(med[m]), 1)[0]) if m.sum() > 3 else None
            rates[f"p{p}_q{q}"] = dict(kind="PL semilog slope of ln(median L - L*) per iteration", window="1e-25 < median < 1e-5",
                                       n_points=int(m.sum()), measured=sl, predicted_local=2 * math.log(1 - 0.2 * th["eta"]),
                                       certified=math.log(1 - th["eta"] * th["mu_certified"]), raw_file=f"raw/{n}.npz:post_stats[:,2]")
        else:
            m = (k >= 15_000) & (k <= 45_000) & np.isfinite(med) & (med > 0)
            sl = float(np.polyfit(np.log(th["eta"] * k[m]), np.log(med[m]), 1)[0]) if m.sum() > 3 else None
            rates[f"p{p}_q{q}"] = dict(kind="KL log-log slope vs t = eta k", window="k in [15000, 45000]", n_points=int(m.sum()),
                                       measured=sl, predicted=-1 / (2 * 0.75 - 1), raw_file=f"raw/{n}.npz:post_stats[:,2]")
floors = {f"p{p}_q{q}_{sch}": S(f"S1_p{p}_q{q}_saddle_{sch}")["floor_median_tail_loss"]
          for p in PS for q in QS for sch in ("fim_const", "fim_rm", "fim_decay", "fim_off", "iso_const", "iso_rm", "sign_const")}
json.dump(dict(rates=rates, floors_median_tail_loss=floors), open(os.path.join(RES, "convergence_rates.json"), "w"), indent=1)

# ============================================================================ theorem checks
fim_like = [n for n in R if C(n)["step"] == "const" and C(n)["sched"] == "const" and C(n)["init"] == "saddle"
            and C(n)["noise"] in ("fim", "iso", "sign")]
checks = []


def row(n, **extra):
    s, th = S(n), T(n)
    return dict(name=n, raw_file=C(n)["raw_file"], p=th["p"], q=th["q"], noise=C(n)["noise"], eta=th["eta"],
                mult=th["noise_mult"], r_frac=th["r_frac"], **extra)


for item, ps, thm in ((1, (2,), "Thm 1"), (2, (4, 6, 8), "Thm 3")):
    ev = []
    for n in fim_like:
        th, s = T(n), S(n)
        if th["p"] not in ps or C(n).get("variant") == "efftime":   # efftime replays the thm-variant trajectories
            continue
        ev.append(row(n, assumptions_met=escape_assumptions(th), bound=th["thm3_tau_bound"], n_censored=s["n_censored"],
                      censored_mean=s["t_escape_censored_mean_lower_bound"], ci95=s["t_escape_censored_mean_ci95"],
                      ratio=s["censored_mean_vs_bound"], verdict=bound_verdict(s, th)))
    met = [e for e in ev if e["assumptions_met"]]
    viol = [e for e in met if e["verdict"] in ("above bound", "censored lower bound exceeds bound")]
    checks.append(dict(item=item, theorem=thm, statement="E[tau | F_N] <= N + c((rho2+rho3)/(eta rho1))^2 (r^2 + K)",
                       assumptions=("Eq. (8) radius, rho3 bounds g^2 on |a|<=r" if item == 1 else
                                    "Lemma 2 radius r <= min(r_lin, r0) (encodes the noise level), rho3 bounds g^2 on |a|<=r"),
                       figure="A2", n_configs=len(ev), n_assumptions_met=len(met),
                       n_met_below_bound=sum(e["verdict"] == "below bound" for e in met), n_met_violations=len(viol),
                       max_ratio_met=max((e["ratio"] for e in met), default=None),
                       verdicts_not_met=sorted({e["verdict"] for e in ev if not e["assumptions_met"]}),
                       verdict=("not tested" if not met else "contradicted as worded" if viol else "supported"), evidence=ev))

sgd = [row(n, init=C(n)["init"], pct_escaped=S(n)["pct_escaped"], gd_escape=T(n)["gd_escape_iters_from_near_init"],
           gd_kind=T(n)["gd_escape_kind"]) for n in R if n.startswith("S1_") and n.endswith("_sgd_const")]
checks.append(dict(item=3, theorem="SGD baseline (motivation, Sec. 1; M_{n+1} = 0 here so SGD = GD)",
                   statement="gradient information alone does not escape degenerate saddles", figure="A2(b), A3(a)",
                   verdict="supported (within budget)" if all(e["pct_escaped"] == 0 for e in sgd if e["init"] == "saddle") else "see evidence",
                   evidence=sorted(sgd, key=lambda e: (e["init"], e["q"], e["p"]))))

ho_ok, ho_not = [], []
for n in fim_like:
    th, s = T(n), S(n)
    e = row(n, r=th["r"], a_crit=th["a_crit"], exact_radius_b0=th["exact_radius_along_b0"], mu=th["mu"],
            mu_attainable=th["flag_mu_attainable"], pct_in_region_at_escape=s["pct_immediate_handover"],
            wilson95=s["pct_in_region_at_escape_wilson95"], pct_handover_at_or_before_escape=s["pct_handover_at_or_before_escape"],
            pct_handover=s["pct_handover"])
    (ho_ok if th["flag_r_ge_acrit"] else ho_not).append(e)
checks.append(dict(item=4, theorem="Thm 2 / Thm 4 / Cor. 1", statement="at the exit time theta_tau lies in a local PL/KL region",
                   assumptions="r >= certified boundary a_c (and mu <= 0.2 for a PL region, pitfall 1)", figure="A3(c)",
                   n_r_ge_acrit=len(ho_ok), min_pct_in_region_at_escape_when_r_ge_acrit=min(e["pct_in_region_at_escape"] for e in ho_ok),
                   mu_not_attainable=[e for e in ho_ok + ho_not if not e["mu_attainable"]], cases_r_below_acrit=ho_not,
                   verdict="supported where r >= a_c" if min(e["pct_in_region_at_escape"] for e in ho_ok) == 100 else "partly supported"))

thr = []
for p in PS:
    for q in QS:
        for noise in ("fim", "iso", "sign"):
            for mult in MULTS:
                n = name_base(p, q, noise, mult=mult)
                thr.append(row(n, noise_threshold_met=T(n)["flag_noise_threshold"], pct_escaped=S(n)["pct_escaped"],
                               pct_handover=S(n)["pct_handover"], pct_in_region_at_escape=S(n)["pct_immediate_handover"],
                               censored_mean=S(n)["t_escape_censored_mean_lower_bound"]))
below = [e for e in thr if not e["noise_threshold_met"] and e["noise"] == "fim"]
checks.append(dict(item=5, theorem="Eq. (35) / Eq. (41)", statement="noise threshold for handover (sufficient condition)",
                   figure="A3(d), A2(d)", n_fim_below_threshold=len(below),
                   n_fim_below_threshold_with_100pct_handover=sum(e["pct_handover"] == 100 for e in below),
                   verdict="threshold is sufficient, not necessary" if any(e["pct_handover"] == 100 for e in below) else "see evidence",
                   evidence=thr))

trade = []
for p in PS:
    r2, r4 = T(name_base(p, 2, "fim")), T(name_base(p, 4, "fim"))
    emp = {}
    for q in QS:
        ok = [m for m in MULTS if S(name_base(p, q, "fim", mult=m))["pct_handover"] >= 95]
        emp[q] = min(ok) if ok else None
    trade.append(dict(p=p, eq35_rhs_PL=r2["eq35_rhs"], eq41_rhs_KL=r4["eq35_rhs"], r_PL=r2["r"], r_KL=r4["r"],
                      mu_PL=r2["mu"], mu_KL=r4["mu"], min_mult_95pct_handover_PL=emp[2], min_mult_95pct_handover_KL=emp[4]))
checks.append(dict(item=6, theorem="Cor. 1 trade-off", statement="entering a PL region needs more noise than entering a KL region",
                   note="mu is not comparable across q (different inequalities); r differs too (r_lin depends on q)",
                   figure="A3(d)", evidence=trade,
                   verdict="not tested cleanly (mu not comparable; empirical minimum multiplier is at the grid floor)"))

thm5 = []
for p in PS:
    for q in QS:
        for noise, var in ((nz, v) for nz in ("fim", "iso", "sign") for v in ("thm", "fixednoise", "efftime")):
            names = [name_base(p, q, noise, eta=e, var=var) for e in ETAS]
            for metric in ("saddle_reentries", "basin_switches"):
                key = "pct_saddle_reentry" if metric == "saddle_reentries" else "pct_basin_switch_after_handover"
                x = [round(S(n)[key] * 200 / 100) for n in names]
                ca = cochran_armitage(x, [200] * 4, np.log(ETAS))
                thm5.append(dict(p=p, q=q, noise=noise, variant=var, metric=("slide-back (paper: saddle re-entry)" if metric == "saddle_reentries"
                                                                 else "basin switch"), etas=list(ETAS), pct=[S(n)[key] for n in names],
                                 last_event_max=[S(n)["last_reentry_max" if metric == "saddle_reentries" else "last_switch_max"] for n in names],
                                 falls_as_eta_falls=bool(ca["z"] is not None and ca["z"] > 0 and ca["p_value"] < 0.05),
                                 rises_as_eta_falls=bool(ca["z"] is not None and ca["z"] < 0 and ca["p_value"] < 0.05),
                                 cochran_armitage_vs_log_eta=ca, files=names))
fim5 = {(v, m): [e for e in thm5 if e["noise"] == "fim" and e["variant"] == v and e["metric"].startswith(m)]
        for v in ("thm", "fixednoise", "efftime") for m in ("slide", "basin")}
trend = {f"{v}/{m}": dict(falls=sum(e["falls_as_eta_falls"] for e in es), rises=sum(e["rises_as_eta_falls"] for e in es), n=len(es))
         for (v, m), es in fim5.items()}
checks.append(dict(item=7, theorem="Thm 5", statement="no slide-back (exponential retention as eta -> 0)",
                   note=("Thm 5 is an eta -> 0 limit. 'fixednoise' (rho1, rho2 fixed) is the small-noise regime of Thm 5; 'thm' rescales "
                         "rho1 ~ eta^-1/2 so the noise per unit time does not shrink; 'efftime' = thm with N*eta fixed. Slide-back = re-entry "
                         "into |a| < r after escape (paper's wording, Thm 6); basin switch = crossing a = 0 after handover."),
                   figure="A5, A5b", fim_trend_counts=trend, evidence=thm5,
                   verdict=("not observed at the eta tested: no Eq. (1) case shows slide-back or basin switch falling as eta falls "
                            "(an eta -> 0 limit is not contradicted by finite-eta data)")
                   if not any(t["falls"] for t in trend.values()) else "partly supported (see fim_trend_counts)"))

t6 = [dict(p=p, q=q, const_floor=S(f"S1_p{p}_q{q}_saddle_fim_const")["floor_median_tail_loss"],
           rm_floor=S(f"S1_p{p}_q{q}_saddle_fim_rm")["floor_median_tail_loss"],
           rm_reentry=S(f"S1_p{p}_q{q}_saddle_fim_rm")["pct_saddle_reentry"],
           rm_in_region_end=S(f"S1_p{p}_q{q}_saddle_fim_rm")["pct_in_region_at_end"],
           rm_escaped=S(f"S1_p{p}_q{q}_saddle_fim_rm")["pct_escaped"], n0=T(f"S1_p{p}_q{q}_saddle_fim_rm")["n0_rm"]) for p in PS for q in QS]
checks.append(dict(item=8, theorem="Thm 6", statement="Robbins-Monro step: convergence to theta_min without re-entry w.h.p.",
                   figure="A4(c), A6", evidence=t6,
                   verdict="partly supported (lower floor than constant step; re-entry before lock-in still occurs)"
                   if all(e["rm_floor"] < e["const_floor"] for e in t6) else "see evidence"))

pl = [rates[f"p{p}_q2"] for p in PS]; kl = [rates[f"p{p}_q4"] for p in PS]
checks.append(dict(item=9, theorem="PL linear / KL sublinear rates; 'rates governed by the degree of degeneracy p'",
                   statement="noise-off diagnostic after certified handover", figure="A4(a,b)",
                   pl_measured=[r["measured"] for r in pl], pl_predicted_local=pl[0]["predicted_local"],
                   kl_measured=[r["measured"] for r in kl], kl_predicted=-2.0, evidence=rates,
                   verdict="PL linear and KL t^-2 supported; asymptotic rate does NOT depend on p"))

checks.append(dict(item=10, theorem="Eq. (1) with constant step", statement="convergence to a local minimum (p.4)",
                   figure="A4(c)", evidence={k: v for k, v in floors.items() if k.endswith(("fim_const", "fim_off"))},
                   verdict="contradicted as worded: constant-step Eq. (1) has a nonzero noise floor at every p"
                   if all(floors[f"p{p}_q{q}_fim_const"] > 1e6 * max(floors[f"p{p}_q{q}_fim_off"], 1e-300) for p in PS for q in QS)
                   else "see evidence"))

cmp_ = []
for p in PS:
    for q in QS:
        e = dict(p=p, q=q)
        for noise in ("fim", "iso", "sign"):
            s = S(name_base(p, q, noise))
            e[noise] = dict(censored_mean=s["t_escape_censored_mean_lower_bound"], ci95=s["t_escape_censored_mean_ci95"],
                            reentry=s["pct_saddle_reentry"], switch=s["pct_basin_switch_after_handover"],
                            in_region_end=s["pct_in_region_at_end"], floor=s["floor_median_tail_loss"], pct_escaped=s["pct_escaped"])
        e["escape_time_ratio_iso_over_fim"] = e["iso"]["censored_mean"] / e["fim"]["censored_mean"]
        e["escape_time_ratio_sign_over_fim"] = e["sign"]["censored_mean"] / e["fim"]["censored_mean"]
        cmp_.append(e)
checks.append(dict(item=11, theorem="Fisher-scaled (Eq. 1) vs isotropic vs constant-magnitude sign noise",
                   statement="FIM-scaled anisotropic noise is the mechanism (Contribution 1)", figure="A2(b), A5, A6", evidence=cmp_,
                   verdict="escape: no advantage for Eq. (1); retention/floor: see evidence"))

divs = [row(n, pct_diverged=S(n)["pct_diverged"]) for n in R if S(n)["pct_diverged"] > 0]
fails = {f: sorted(n for n in R if T(n).get(f) is False) for f in FLAGS}
checks.append(dict(item=12, theorem="Divergences and failed assumption flags", figure="table_all_configs.csv",
                   divergences=divs, n_configs=len(R), flag_failure_counts={f: len(v) for f, v in fails.items()},
                   flag_failures=fails, verdict=f"{len(divs)} configurations with divergence"))
json.dump(checks, open(os.path.join(RES, "theorem_checks.json"), "w"), indent=1, default=float)

# ============================================================================ A1: saddle geometry
fig, axs = plt.subplots(2, 3, figsize=(11, 6.4))
aa = np.linspace(-1.3, 1.3, 1201)
for j, q in enumerate(QS):
    ax = axs[0, j]
    for p in PS:
        th = T(name_base(p, q, "fim"))
        ax.plot(aa, loss(p, q, aa, 0.0), color=CP[p], label=f"p={p}")
        ax.axvline(th["r"], color=CP[p], ls="--", lw=0.8); ax.axvline(-th["r"], color=CP[p], ls="--", lw=0.8)
        ax.plot([th["a_crit"], -th["a_crit"]], [loss(p, q, th["a_crit"], 0.0)] * 2, "o", color=CP[p], ms=4)
    ax.set(ylim=(0, 0.2), xlabel="a (b = 0)", ylabel="L(a, 0)",
           title=f"({'ab'[j]}) q={q} ({'PL' if q == 2 else 'KL'}): profile; dashed |a|=r, dot = certified a_c")
    ax.legend()
ax = axs[0, 2]
ag = np.geomspace(1e-3, 0.5, 200)
for p in PS:
    ax.loglog(ag, -0.1 * np.expm1(2 * np.log1p(-ag ** p)), color=CP[p], label=f"p={p}")  # = L(0)-L(a,0) without cancellation
    ax.loglog(ag, 0.2 * ag ** p, color=CP[p], ls=":", lw=0.8)
ax.set(ylim=(1e-25, 1), xlabel="a", ylabel="L(0) − L(a, 0)", title="(c) depth of the saddle along u, q=2 (dotted: 0.2 a^p)")
ax.legend()
ax = axs[1, 0]
x = np.arange(len(PS))
for j, q in enumerate(QS):
    ev = [T(name_base(p, q, "fim"))["mu_deg"] if p == 2 else 0.0 for p in PS]  # a-direction Hessian eigenvalue
    ax.bar(x + (j - 0.5) * 0.35, ev, 0.33, color=["#eb6834", "#2a78d6"][j], label=f"a-direction, q={q}")
ax.axhline(0.2, color="0.3", ls="--", lw=0.9, label="b-direction (0.2, all p, q)")
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(x, [f"p={p}" for p in PS]); ax.set(ylabel="Hessian eigenvalue at (0,0)",
                                                 title="(d) Hessian at the saddle: strict only at p=2")
ax.legend()
ax = axs[1, 1]
for q, mk in ((2, "o"), (4, "s")):
    ax.semilogy(PS, [T(name_base(p, q, "fim"))["noise_step_at_zero_grad"] for p in PS], mk + "-", color="#eb6834",
                label=f"noise step ηρ₁/ρ₂, q={q}")
    ax.semilogy(PS, [T(name_base(p, q, "fim"))["r"] for p in PS], mk + "--", color="0.3", label=f"escape radius r, q={q}")
ax.set(xlabel="p", ylabel="length in parameter space", title="(e) noise step at zero gradient vs escape radius")
ax.legend()
ax = axs[1, 2]
for q, mk in ((2, "o"), (4, "s")):
    for rf, ls in ((0.5, "-"), (1.0, ":")):
        ax.semilogy(PS, [T(name_base(p, q, "fim", rf=rf))["mu"] for p in PS], mk + ls, color=["#eb6834", "#2a78d6"][q == 4],
                    label=f"q={q}, r={rf}·r_lin")
ax.axhline(0.2, color="k", lw=0.8, ls="--"); ax.text(2, 0.24, "PL ceiling 0.2 (b-direction)", fontsize=6.5)
ax.set(xlabel="p", ylabel="μ from Thm 2 / Thm 4 / Cor. 1", title="(f) handover constant μ vs p")
ax.legend()
save(fig, "A1_saddle_geometry")

fig, axs = plt.subplots(2, 4, figsize=(13, 6.2), sharey=True)
for i, q in enumerate(QS):
    for j, p in enumerate(PS):
        ax = axs[i, j]
        n = f"S1_p{p}_q{q}_saddle_fim_const"
        d, th = raw(n), T(n)
        A_, B_ = np.meshgrid(np.linspace(-1.3, 1.3, 260), np.linspace(-1.0, 1.0, 200))
        ax.contourf(A_, B_, np.log10(loss(p, q, A_, B_) + 1e-6), levels=14, cmap="Greys_r", alpha=0.4)
        for s_ in (-1, 1):
            ax.axvline(s_ * th["a_crit"], color="#e34948", lw=1); ax.axvline(s_ * th["r"], color="k", ls="--", lw=0.8)
        G, KA, KB = d["grid"], d["keep_a"].astype(float), d["keep_b"].astype(float)
        sw = d["e0_basin_switches"][:20] > 0
        m = G <= 1500
        for idx, c in ((np.flatnonzero(~sw), "#eb6834"), (np.flatnonzero(sw), "#1baf7a")):
            if len(idx):
                ax.plot(KA[m, idx[0]], np.clip(KB[m, idx[0]], -1, 1), color=c, lw=0.6)
        ax.plot(0, 0, "o", color="#2a78d6", ms=4)
        ax.set(xlim=(-1.3, 1.3), ylim=(-1, 1), title=f"p={p}, q={q}: r={th['r']:.3f}, a_c={th['a_crit']:.3f}")
        if i == 1: ax.set_xlabel("a")
        if j == 0: ax.set_ylabel("b")
fig.suptitle("A1b. Contours of log L; red: certified boundary |a|=a_c; dashed: escape radius |a|=r; orange: an Eq. (1) "
             "trajectory with no basin switch; green: one that switches basin (first 1,500 iterations; b clipped to ±1 — at p=2 the noise drives |b| above 1)", fontsize=8)
save(fig, "A1b_contours_trajectories")

# ============================================================================ A2: empirical vs theoretical escape
fig, axs = plt.subplots(2, 3, figsize=(12, 7))
ax = axs[0, 0]
for n in fim_like:
    th, s, c = T(n), S(n), C(n)
    if c["sweep"] == "S2" and c["variant"] == "efftime":
        continue
    met = escape_assumptions(th)
    mk = "^" if s["n_censored"] else "o"
    ax.scatter(th["thm3_tau_bound"], s["t_escape_censored_mean_lower_bound"], marker=mk, s=14,
               facecolor=CO[c["noise"]] if met else "none", edgecolor=CO[c["noise"]], lw=0.7)
lim = [1, 1e12]
ax.plot(lim, lim, "k-", lw=0.8)
ax.set(xscale="log", yscale="log", xlim=(3, 1e12), ylim=(1, 1e5), xlabel="Thm 1 / Thm 3 bound",
       ylabel="measured mean escape time (censored ▲ = lower bound)", title="(a) measured vs bound; filled = assumptions met")
for k, v in CO.items():
    if k != "none":
        ax.scatter([], [], color=v, s=14, label=LAB[k])
ax.legend(loc="lower right")
ax = axs[0, 1]
for noise in ("fim", "iso", "sign"):
    for q, ls in ((2, "-"), (4, "--")):
        y = [S(name_base(p, q, noise))["t_escape_censored_mean_lower_bound"] for p in PS]
        ax.semilogy(PS, y, "o" + ls, color=CO[noise], ms=3, label=f"{LAB[noise]}, q={q}" if q == 2 else None)
for q, ls in ((2, "-"), (4, "--")):
    ax.semilogy(PS, [T(name_base(p, q, "fim"))["thm3_tau_bound"] for p in PS], "k" + ls, lw=1, label="bound" if q == 2 else None)
    ax.semilogy(PS, [T(name_base(p, q, "fim"))["random_walk_escape_estimate"] for p in PS], ":", color="0.5", lw=1,
                label="(r / step)² random walk" if q == 2 else None)
    ax.semilogy(PS, [T(name_base(p, q, "fim"))["gd_escape_iters_from_near_init"] for p in PS], "s" + ls, color=CO["none"], ms=3,
                label="SGD from (0.01,0.01) (p≥6: ODE est.)" if q == 2 else None)
ax.axhline(50_000, color="0.6", lw=0.6)
ax.set(xlabel="p", ylabel="escape iterations", title="(b) escape time vs p (solid q=2, dashed q=4)")
ax.legend(fontsize=6)
for ax, axis, vals, lab_ in ((axs[0, 2], "eta", ETAS, "η"), (axs[1, 0], "mult", MULTS, "noise multiplier"),
                             (axs[1, 1], "rf", RFS, "r / r_lin")):
    for p in PS:
        for q, ls in ((2, "-"), (4, "--")):
            names = [name_base(p, q, "fim", **{axis: v}) for v in vals]
            ax.loglog(vals, [S(n)["t_escape_censored_mean_lower_bound"] for n in names], "o" + ls, color=CP[p], ms=3,
                      label=f"p={p}" if q == 2 else None)
            ax.loglog(vals, [T(n)["thm3_tau_bound"] for n in names], ls, color=CP[p], lw=0.6, alpha=0.5)
    ax.set(xlabel=lab_, ylabel="escape iterations (Eq. 1)", ylim=(1, 1e8),
           title=f"({'cde'[('eta', 'mult', 'rf').index(axis)]}) vs {lab_}; thin = bound" + (" (same trajectories)" if axis == "rf" else ""))
    ax.set_xticks(vals, [f"{v:g}" for v in vals]); ax.minorticks_off()
    ax.legend()
ax = axs[1, 2]
for i, p in enumerate(PS):
    ys = [S(n)["censored_mean_vs_bound"] for n in fim_like if T(n)["p"] == p and escape_assumptions(T(n))
          and not (C(n)["sweep"] == "S2" and C(n)["variant"] == "efftime")]
    ax.scatter(np.full(len(ys), p) + np.random.default_rng(0).uniform(-0.25, 0.25, len(ys)), ys, s=8, color=CP[p])
ax.axhline(1, color="k", lw=0.8)
ax.set(yscale="log", xticks=PS, xlabel="p", ylabel="measured / bound", title="(f) measured / bound, assumptions met (all noises)")
save(fig, "A2_escape_vs_theory")

# ============================================================================ A3: success-rate heatmaps
fig, axs = plt.subplots(2, 4, figsize=(17, 6.6))
pcols = [f"p={p}" for p in PS]
for j, q in enumerate(QS):
    rows, M = [], []
    for init in ("saddle", "near"):
        for lab, noise in (("sgd_const", "none"), ("fim_const", "fim"), ("iso_const", "iso"), ("sign_const", "sign")):
            rows.append(f"{LAB[noise]}, {init}")
            M.append([S(f"S1_p{p}_q{q}_{init}_{lab}")["pct_escaped"] for p in PS])
    heat(axs[0, j], np.array(M), rows, pcols, f"(a) escape %, q={q}")
    rows, M = [], []
    for init in ("saddle", "near"):
        for lab in ("fim_const", "fim_rm", "fim_decay", "fim_off"):
            rows.append(f"{lab}, {init}")
            M.append([S(f"S1_p{p}_q{q}_{init}_{lab}")["pct_descent"] for p in PS])
    heat(axs[1, j], np.array(M), rows, pcols, f"(b) Eq. (1) descent % (L < 0.09), q={q}")
    M = np.array([[S(name_base(p, q, "fim", rf=rf))["pct_immediate_handover"] for p in PS] for rf in RFS])
    heat(axs[0, 2 + j], M, [f"r={rf}·r_lin" for rf in RFS], pcols, f"(c) in region at escape %, q={q}")
    M = np.array([[S(name_base(p, q, "fim", mult=m))["pct_escaped"] for p in PS] for m in MULTS])
    heat(axs[1, 2 + j], M, [f"noise ×{m}" for m in MULTS], pcols, f"(d) Eq. (1) escape %, q={q}")
fig.suptitle("A3. Percent of 200 trajectories (100% of 200 has a 95% Wilson interval [98.1, 100]); Eq. (1) unless labelled", fontsize=8.5)
save(fig, "A3_success_rates")

# ============================================================================ A4: convergence
fig, axs = plt.subplots(2, 3, figsize=(12, 7))
for p in PS:
    d = raw(f"S1_p{p}_q2_saddle_fim_off")
    k, med = d["post_grid"], d["post_stats"][:, 2]
    axs[0, 0].semilogy(k, med, color=CP[p], label=f"p={p}: slope {rates[f'p{p}_q2']['measured']:.5f}")
    d = raw(f"S1_p{p}_q4_saddle_fim_off")
    k, med = d["post_grid"], d["post_stats"][:, 2]
    axs[0, 1].loglog(0.01 * k + 0.01, med, color=CP[p], label=f"p={p}: slope {rates[f'p{p}_q4']['measured']:.2f}")
kk = np.arange(0, 12001, 100)
axs[0, 0].semilogy(kk, 1e-3 * (1 - 0.002) ** (2 * kk), "k:", label="predicted 2 ln(1−0.2η) = −0.00400")
axs[0, 0].set(xlim=(0, 12000), ylim=(1e-30, 1), xlabel="k = n − τ_h", ylabel="median L − L*",
              title="(a) PL, noise off after handover (diagnostic)")
tt = np.geomspace(10, 500, 20)
axs[0, 1].loglog(tt, 1e-5 * (tt / 10) ** -2, "k:", label="predicted t⁻²")
axs[0, 1].set(xlabel="t = η k", ylabel="median L − L*", title="(b) KL, noise off after handover (diagnostic)")
for ax in axs[0, :2]:
    ax.legend()
ax = axs[0, 2]
sch = ("fim_const", "fim_rm", "fim_decay", "fim_off", "iso_const")
w = 0.15
for i, s_ in enumerate(sch):
    for q, hatch in ((2, ""), (4, "///")):
        ys = [max(floors[f"p{p}_q{q}_{s_}"], 1e-35) for p in PS]
        ax.bar(np.arange(4) + (i - 2) * w + (q == 4) * w / 2, ys, w / 2, hatch=hatch, label=s_ if q == 2 else None,
               color=["#eb6834", "#eda100", "#e34948", "#6250d6", "#1baf7a"][i])
ax.set(yscale="log", xticks=range(4), xticklabels=pcols, ylabel="floor (median tail loss)",
       title="(c) loss floor by p and schedule (hatched q=4)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
for j, q in enumerate(QS):
    ax = axs[1, j]
    for s_, ls in (("fim_const", "-"), ("fim_rm", "--"), ("fim_decay", "-."), ("fim_off", ":")):
        d = raw(f"S1_p4_q{q}_saddle_{s_}")
        ax.loglog(d["post_grid"] + 1, d["post_stats"][:, 2], ls, color="#eb6834", label=s_)
    ax.set(ylim=(1e-12, 0.3), xlabel="k + 1 since handover", ylabel="median L − L*",
           title=f"({'de'[j]}) schedules, p=4, q={q} (decay/off are diagnostics)")
    ax.legend()
ax = axs[1, 2]
for p in PS:
    for noise, ls in (("fim", "-"), ("iso", "--")):
        ax.loglog(ETAS, [S(name_base(p, 2, noise, eta=e))["floor_median_tail_loss"] for e in ETAS], "o" + ls, color=CP[p],
                  ms=3, label=f"p={p}, {LAB[noise]}")
ax.set(xlabel="η (theorem noise)", ylabel="floor (median tail loss), q=2", title="(f) floor vs η (solid Eq. 1, dashed isotropic)")
ax.set_xticks(ETAS, [f"{e:g}" for e in ETAS]); ax.minorticks_off()
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=6)
save(fig, "A4_convergence")

# ============================================================================ A5: slide-back heatmaps
for metric, key, fname in (("slide-back = saddle re-entry (paper)", "pct_saddle_reentry", "A5_slideback_reentry"),
                           ("basin switch", "pct_basin_switch_after_handover", "A5b_basin_switch")):
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.2))
    for i, q in enumerate(QS):
        for j, noise in enumerate(("fim", "iso")):
            M = np.array([[S(name_base(p, q, noise, eta=e))[key] for p in PS] for e in ETAS])
            heat(axs[i, j], M, [f"η={e}" for e in ETAS], pcols, f"{LAB[noise]}, q={q}: vs η")
            M = np.array([[S(name_base(p, q, noise, mult=m))[key] for p in PS] for m in MULTS])
            heat(axs[i, 2 + j], M, [f"noise ×{m}" for m in MULTS], pcols, f"{LAB[noise]}, q={q}: vs noise multiplier")
    fig.suptitle(f"{fname.split('_')[0]}. {metric} after escape, % of 200 trajectories (η rows: theorem noise, 50k iterations; "
                 "η=0.01 and ×1 rows are the base point)", fontsize=8.5)
    save(fig, fname)

# ============================================================================ A6: after-handover schedules
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
sch = ("fim_const", "fim_rm", "fim_decay", "fim_off")
xl = [f"p{p}q{q}" for q in QS for p in PS]
for ax, key, title in ((axs[0], "pct_saddle_reentry", "slide-back (saddle re-entry) %"),
                       (axs[1], "pct_in_region_at_end", "in certified region at end %"),
                       (axs[2], "floor_median_tail_loss", "final loss (median, last 10k it.)")):
    for i, s_ in enumerate(sch):
        ys = [S(f"S1_p{p}_q{q}_saddle_{s_}")[key] for q in QS for p in PS]
        if key.startswith("floor"):
            ys = [max(y, 1e-35) for y in ys]
        ax.bar(np.arange(8) + (i - 1.5) * 0.2, ys, 0.19, label=s_, color=["#eb6834", "#eda100", "#e34948", "#6250d6"][i])
    ax.set_xticks(range(8), xl, fontsize=6.5); ax.set_title(title)
    if key.startswith("floor"):
        ax.set_yscale("log")
axs[0].legend()
save(fig, "A6_after_handover_schedules")

# ============================================================================ console digest for the report
for c in checks:
    print(f"\n#{c['item']} {c['theorem']}: {c['verdict']}")
    for k in ("n_configs", "n_assumptions_met", "n_met_below_bound", "n_met_violations", "max_ratio_met", "verdicts_not_met",
              "n_r_ge_acrit", "min_pct_in_region_at_escape_when_r_ge_acrit", "n_fim_below_threshold",
              "n_fim_below_threshold_with_100pct_handover", "fim_trend_counts", "pl_measured", "kl_measured",
              "flag_failure_counts"):
        if k in c:
            print(f"   {k}: {c[k]}")
print("\nrates:", json.dumps(rates, indent=0)[:1500])
