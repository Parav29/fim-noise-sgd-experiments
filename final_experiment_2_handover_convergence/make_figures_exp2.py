"""Figures + convergence numbers for Final Experiment 2. Reads raw/*.npz and summary.json from run_experiment2.py."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_experiment2 import loss, ETA, N0_RM, N_ITER

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
C = dict(sgd="#2a78d6", fim="#eb6834", iso="#1baf7a", off="#6250d6")  # validated categorical slots 1,2,3,7
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})
S = json.load(open(os.path.join(HERE, "summary.json")))
ld = lambda n: np.load(os.path.join(RAW, n + ".npz"))


def band(ax, x, st, c, label, ls="-", lw=1.6):
    m = np.isfinite(st[:, 2])
    ax.fill_between(x[m], st[m, 0], st[m, 4], color=c, alpha=0.15, lw=0)
    ax.plot(x[m], st[m, 5], color=c, ls=ls, lw=lw, label=label)          # mean
    ax.plot(x[m], st[m, 2], color=c, ls=":", lw=0.9)                       # median


def accumulated_step(k, t0):
    """sum of alpha_n for n = t0 .. t0+k-1 under the Robbins-Monro schedule."""
    n = np.arange(int(t0), int(t0) + int(k.max()) + 1)
    cs = np.concatenate([[0.0], np.cumsum(ETA * N0_RM / (N0_RM + n))])
    return cs[k.astype(int)]


def tail_slope(x, y, lo, hi, logx=False):
    m = (x >= lo) & (x <= hi) & np.isfinite(y) & (y > 0)
    xx = np.log(x[m]) if logx else x[m]
    return float(np.polyfit(xx, np.log(y[m]), 1)[0]) if m.sum() > 3 else None


conv = {}
for q in (2, 4):
    T = S[f"q{q}"]["theory"]
    r, ac, mu = T["r"], T["a_crit"], T["mu"]
    tag = "PL" if q == 2 else "KL"
    base = f"q{q}_saddle"
    fc, fr, ic, ir, sc, off = (ld(f"{base}_{x}") for x in ("fim_const", "fim_rm", "iso_const", "iso_rm", "sgd_const",
                                                           "fim_const_noiseoff"))
    G = fc["grid"]
    x1 = G + 1.0
    fig = plt.figure(figsize=(21, 9.5))

    # ---- Panel A: 3D landscape
    ax = fig.add_subplot(2, 4, 1, projection="3d")
    aa, bb = np.meshgrid(np.linspace(-1.35, 1.35, 181), np.linspace(-1.0, 1.0, 121))
    zc = 0.35
    Z = np.minimum(loss(q, aa, bb), zc)
    ax.plot_surface(aa, bb, Z, cmap="viridis", alpha=0.55, lw=0, rstride=2, cstride=2)
    for s_ in (-1, 1):
        for rad, cc, ls in ((r, "k", "--"), (ac, "#e34948", "-")):
            yb = np.linspace(-1, 1, 50)
            ax.plot(np.full_like(yb, s_ * rad), yb, np.minimum(loss(q, s_ * rad, yb), zc), color=cc, ls=ls, lw=1.2)
    kA, kB = fc["keep_a"], fc["keep_b"]
    for j in range(3):
        m = G <= 3000
        ax.plot(kA[m, j], kB[m, j], np.minimum(loss(q, kA[m, j], kB[m, j]), zc) + 0.004, color=C["fim"], lw=0.9)
    ax.scatter([0], [0], [0.1], color=C["sgd"], s=40, depthshade=False, label="saddle = start; SGD stays")
    ax.scatter([-1, 1], [0, 0], [0, 0], color="k", marker="*", s=70, depthshade=False, label="minima")
    ax.set(xlabel="a", ylabel="b", zlabel="L (clipped)", title=f"A. Loss landscape, q={q} ({tag} destination)")
    ax.view_init(32, -62)
    ax.legend(loc="upper left", fontsize=7)

    # ---- Panel B: contour + trajectories
    ax = fig.add_subplot(2, 4, 2)
    aa2, bb2 = np.meshgrid(np.linspace(-1.35, 1.35, 401), np.linspace(-1.0, 1.0, 301))
    ax.contour(aa2, bb2, np.log10(loss(q, aa2, bb2) + 1e-6), levels=18, cmap="viridis", linewidths=0.6)
    for s_ in (-1, 1):
        ax.axvspan(s_ * ac, s_ * 1.35, color="#e34948", alpha=0.05, lw=0)
        ax.axvline(s_ * ac, color="#e34948", lw=1.3, label=(f"certified {tag} boundary |a|={ac:.3f}" if s_ > 0 else None))
        ax.axvline(s_ * r, color="k", ls="--", lw=1.1, label=(f"theory escape radius |a|=r={r:.3f}" if s_ > 0 else None))
    te, th_ = fc["t_esc"], fc["t_handover"]
    for j in range(3):
        m = G <= 3000
        ax.plot(kA[m, j], kB[m, j], color=C["fim"], lw=0.7, alpha=0.9, label=("Eq. (1) trajectories (3 of 200)" if j == 0 else None))
        ie, ih = np.searchsorted(G, te[j]), np.searchsorted(G, th_[j])
        ax.scatter(kA[ih, j], kB[ih, j], marker="o", s=28, facecolor="white", edgecolor="#e34948", zorder=5,
                   label=("handover point (first entry)" if j == 0 else None))
        ax.scatter(kA[ie, j], kB[ie, j], marker="x", s=30, color="k", zorder=5, label=("escape point (|a| >= r)" if j == 0 else None))
    nA, nB = ld(f"q{q}_near_sgd_const")["keep_a"], ld(f"q{q}_near_sgd_const")["keep_b"]
    ax.plot(nA[:, 0], nB[:, 0], color=C["sgd"], lw=2, label="SGD from saddle and from (0.01,0.01): no visible motion")
    ax.scatter([0], [0], color=C["sgd"], s=25, zorder=6)
    ax.scatter([-1, 1], [0, 0], color="k", marker="*", s=60, zorder=6)
    ax.set(xlim=(-1.35, 1.35), ylim=(-1.0, 1.0), xlabel="a (escape direction u)", ylabel="b",
           title="B. Contours of log L, saddle start, first 3,000 iterations")
    ax.legend(fontsize=6, loc="lower center")

    # ---- Panel C: loss gap vs iteration
    ax = fig.add_subplot(2, 4, 3)
    band(ax, x1, fc["stat_loss"], C["fim"], "Eq. (1), constant step")
    band(ax, x1, ic["stat_loss"], C["iso"], "isotropic Gaussian, variance-matched")
    band(ax, x1, off["stat_loss"], C["off"], "Eq. (1), noise off after handover (diagnostic)", ls="--")
    band(ax, x1, sc["stat_loss"], C["sgd"], "SGD (= GD here)")
    med_h, med_e = np.median(th_), np.median(te)
    settle = np.percentile(np.where(fc["last_region_exit"] >= 0, fc["last_region_exit"], fc["t_handover"]), 95)
    ax.axvline(med_h + 1, color="#e34948", lw=1.1, label=f"median handover τ_h = {med_h:.0f}")
    ax.axvline(med_e + 1, color="k", ls="--", lw=1.1, label=f"median escape τ_esc = {med_e:.0f}")
    ax.axvspan(1, med_h + 1, color="0.85", alpha=0.4, lw=0)
    ax.axvspan(med_h + 1, settle + 1, color="#e34948", alpha=0.07, lw=0)
    ax.text(1.3, 3e-6, "saddle\nphase", fontsize=7); ax.text(med_h + 1.5, 3e-6, "handover\ntransient", fontsize=7)
    ax.text(settle + 1, 3e-6, f" no region exits after n={settle:.0f}\n (95th pct. of last exit)", fontsize=7)
    ax.set(xscale="log", yscale="log", ylim=(1e-12, 0.3), xlabel="iteration n + 1", ylabel="loss gap L(θ_n) − L* (L* = 0)",
           title="C. Loss gap (mean solid, median dotted, band 5–95%, 200 traj.)")
    ax.legend(fontsize=6.5, loc="lower left")

    # ---- Panel D: PL/KL ratio
    ax = fig.add_subplot(2, 4, 4)
    band(ax, x1, fc["stat_ratio"], C["fim"], f"measured R_{tag}(θ_n), Eq. (1)")
    band(ax, x1, fc["stat_ratio_lb"], "#e34948", "certified pointwise lower bound ψ(θ_n)", ls="--")
    ax.axhline(mu, color="k", lw=1.1, label=f"μ = {mu:.4f} (Thm 4 / Cor. 1 at r)")
    ax.axvline(med_h + 1, color="#e34948", lw=0.8)
    ax.set(xscale="log", yscale="log", ylim=(1e-6, 1e2), xlabel="iteration n + 1",
           ylabel=("R_PL = ‖∇L‖² / 2(L−L*)" if q == 2 else "R_KL = ‖∇L‖ / (L−L*)^{3/4}"),
           title=f"D. Direct {tag} check: ψ ≥ μ ⇔ θ_n in certified region")
    ax.legend(fontsize=6.5, loc="lower right")

    # ---- Panel E1: semilog since handover
    ax = fig.add_subplot(2, 4, 5)
    k = fc["post_grid"].astype(float)
    for d_, c_, lab, ls in ((fc, C["fim"], "Eq. (1), constant step", "-"), (ic, C["iso"], "isotropic, constant step", "-"),
                            (off, C["off"], "Eq. (1), noise off after handover", "--")):
        band(ax, k, d_["post_stats"], c_, lab, ls=ls)
    L0 = np.nanmedian(off["post_stats"][0, 2])
    if q == 2:
        ax.plot(k, L0 * (1 - ETA * mu) ** k, color="k", ls=":", lw=1.1, label="certified PL rate (1 − ημ)^k")
        ax.plot(k, L0 * (1 - 0.2 * ETA) ** (2 * k), color="0.4", ls="-.", lw=1.0, label="local rate (1 − ηλ_min)^{2k}, λ_min = 0.2")
    ax.set(yscale="log", xlim=(0, 6000), ylim=(1e-14, 0.3), xlabel="iterations since handover k = n − τ_h",
           ylabel="L(θ) − L*", title="E. Convergence after handover (semilog)")
    ax.legend(fontsize=6.5, loc="lower left")

    # ---- Panel E2: log-log vs accumulated step size
    ax = fig.add_subplot(2, 4, 6)
    for d_, c_, lab, ls, rm in ((fc, C["fim"], "Eq. (1), constant", "-", False), (fr, C["fim"], "Eq. (1), Robbins–Monro", "--", True),
                                (ic, C["iso"], "isotropic, constant", "-", False), (ir, C["iso"], "isotropic, Robbins–Monro", "--", True),
                                (off, C["off"], "Eq. (1), noise off (diag.)", "-.", False)):
        t = accumulated_step(k, np.median(d_["t_handover"])) if rm else ETA * k
        band(ax, t + ETA, d_["post_stats"], c_, lab, ls=ls)
    if q == 4:
        tt = np.geomspace(1, 500, 50)
        ax.plot(tt, 3e-3 * tt ** -2.0, color="k", ls=":", lw=1.1, label="KL(κ=3/4) prediction ∝ t^{−1/(2κ−1)} = t^{−2}")
    ax.set(xscale="log", yscale="log", ylim=(1e-14, 0.3), xlabel="accumulated step  t = Σ α_n since handover",
           ylabel="L(θ) − L*", title="E′. Same, log-log vs accumulated step (noise floor vs decay)")
    ax.legend(fontsize=6.5, loc="lower left")

    # ---- Panel F: distances
    ax = fig.add_subplot(2, 4, 7)
    band(ax, x1, fc["stat_abs_a"], C["fim"], "|a_n| (displacement along u)")
    band(ax, x1, fc["stat_abs_b"], C["sgd"], "|b_n| (flat direction, λ = 0.2)")
    band(ax, x1, fc["stat_dist_min"], "#e34948", "distance to destination minimum", ls="--")
    band(ax, x1, fc["stat_noise_step"], "0.45", "injected step ‖α_n Υ_n‖", ls="-.")
    ax.axhline(r, color="k", ls="--", lw=1.0, label=f"escape radius r = {r:.3f}")
    ax.axhline(ac, color="#e34948", lw=1.0, label=f"certified boundary a_c = {ac:.3f}")
    rs = S[f"q{q}"]["runs"][f"{base}_fim_const"]
    ax.set(xscale="log", yscale="log", ylim=(1e-4, 3), xlabel="iteration n + 1", ylabel="distance in parameter space",
           title=f"F. Retention: immediate {rs['pct_immediate_handover']:.0f}%, ≥1 region exit {rs['pct_any_region_exit']:.0f}%, "
                 f"basin switch {rs['pct_basin_switch_after_handover']:.0f}%")
    ax.legend(fontsize=6.5, loc="lower left")

    # ---- Panel G: outcome summary
    ax = fig.add_subplot(2, 4, 8)
    names = [("sgd_const", "SGD\nconst"), ("sgd_rm", "SGD\nRM"), ("fim_const", "Eq.(1)\nconst"), ("fim_rm", "Eq.(1)\nRM"),
             ("iso_const", "iso\nconst"), ("iso_rm", "iso\nRM")]
    metrics = [("pct_escaped", "escaped"), ("pct_immediate_handover", "in region at escape"),
               ("pct_any_region_exit", "≥1 region exit after entry"), ("pct_basin_switch_after_handover", "basin switch (slide-back)"),
               ("settled", "no region exit after n = 5,000")]
    w = 0.16
    for mi, (key, lab) in enumerate(metrics):
        vals = []
        for nm, _ in names:
            if key == "settled":
                d_ = ld(f"{base}_{nm}")
                ok = d_["t_handover"] >= 0
                vals.append(100 * np.mean(ok & (d_["last_region_exit"] < 5000)))
            else:
                vals.append(S[f"q{q}"]["runs"][f"{base}_{nm}"][key])
        ax.bar(np.arange(len(names)) + (mi - 2) * w, vals, w * 0.9, label=lab,
               color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"][mi])
    ax.set_xticks(np.arange(len(names)), [n for _, n in names])
    ax.set(ylabel="% of 200 trajectories", ylim=(0, 105), title="G. Outcomes, saddle start (near-saddle start in summary.json)")
    ax.legend(fontsize=6.5, loc="upper left", ncol=2)

    fig.suptitle(f"Final Experiment 2 — q = {q} ({tag} destination): degenerate saddle (p = 4) → escape → certified {tag} region → retention → convergence.  "
                 f"η = {ETA}, ρ₁ = {T['rho1']:.4g}, ρ₂ = ρ₃ = {T['rho2']:.4g} (from Eq. 31/35 at r = r_lin/2), 200 trajectories, 50,000 iterations",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"fig_exp2_q{q}_{tag}.png"), dpi=170)
    plt.close(fig)

    # ---- convergence numbers
    ms_off = off["post_stats"][:, 2]
    conv[f"q{q}"] = dict(
        noiseoff_semilog_slope_k_2000_6000=tail_slope(k, ms_off, 2000, 6000),
        noiseoff_loglog_slope_k_2000_45000=tail_slope(k, ms_off, 2000, 45000, logx=True),
        noiseoff_loglog_slope_k_15000_45000=tail_slope(k, ms_off, 15000, 45000, logx=True),
        certified_pl_rate_ln_1_minus_eta_mu=(float(np.log(1 - ETA * mu)) if q == 2 else None),
        local_b_rate_ln=float(2 * np.log(1 - 0.2 * ETA)),
        fim_const_floor_median_last10k=rs["tail_mean_loss_pct"][2],
        iso_const_floor_median_last10k=S[f"q{q}"]["runs"][f"{base}_iso_const"]["tail_mean_loss_pct"][2],
        fim_rm_loglog_slope_vs_t_late=tail_slope(accumulated_step(k, np.median(fr["t_handover"])) + ETA, fr["post_stats"][:, 2], 5, 1e9, logx=True),
        iso_rm_loglog_slope_vs_t_late=tail_slope(accumulated_step(k, np.median(ir["t_handover"])) + ETA, ir["post_stats"][:, 2], 5, 1e9, logx=True),
        fim_const_floor_vs_time=[float(np.nanmedian(fc["post_stats"][(k >= lo) & (k < hi), 2])) for lo, hi in ((5000, 15000), (15000, 30000), (30000, 49000))],
        settled_after_5000_pct={nm: float(100 * np.mean((ld(f"{base}_{nm}")["t_handover"] >= 0) & (ld(f"{base}_{nm}")["last_region_exit"] < 5000)))
                                for nm in ("fim_const", "fim_rm", "iso_const", "iso_rm")},
        max_last_region_exit={nm: int(ld(f"{base}_{nm}")["last_region_exit"].max()) for nm in ("fim_const", "fim_rm", "iso_const", "iso_rm")},
    )
json.dump(conv, open(os.path.join(HERE, "convergence_numbers.json"), "w"), indent=1)
print(json.dumps(conv, indent=1))
