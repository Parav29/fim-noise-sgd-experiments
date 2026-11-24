"""Figures + numbers for "new experiments result/" (E1-E5). Reads raw/*.json written by run_new_experiments.sh."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "new experiments result"
R = os.path.join(D, "raw")
LN10 = np.log(10.0)
ESC = LN10 - 0.1          # escape = first full-train eval with L < ln10 - 0.1 (~2.20)
SEEDS = [0, 1, 2]
C_SGD, C_FIM = "#2a78d6", "#eb6834"   # validated categorical slots 1-2
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


def load(name):
    return [json.load(open(os.path.join(R, f"{name}_seed{s}.json"))) for s in SEEDS]


def curves(runs, key):
    it = np.array(runs[0][key]["eval_iters"])
    L = np.array([r[key]["eval_loss"] for r in runs])
    return it, L


def escape_iter(it, l):
    i = np.flatnonzero(l < ESC)
    return int(it[i[0]]) if len(i) else None


def slide_back(it, l):
    """After escape: does L ever come back to the plateau band (>= ESC)? Also max rise above running min."""
    t = escape_iter(it, l)
    if t is None:
        return None
    post = l[it >= t]
    return dict(returned=bool((post[1:] >= ESC).any()), max_post=float(post.max()),
                max_rise=float((post - np.minimum.accumulate(post)).max()))


def slopes(it, l, t0, wins=((0, 1500), (1500, 4000), (4000, 9000), (9000, 14000), (14000, 99999))):
    """Least-squares slope of ln L per 1000 iterations in windows measured from the escape iteration t0."""
    out = []
    for a, b in wins:
        m = (it >= t0 + a) & (it < t0 + b)
        if m.sum() >= 3:
            out.append((a, min(b, int(it[-1] - t0)), float(np.polyfit(it[m] / 1000, np.log(l[m]), 1)[0])))
    return out


def band(ax, it, L, c, label, log=False):
    Y = np.log(L) if log else L
    ax.fill_between(it, Y.min(0), Y.max(0), color=c, alpha=0.2, lw=0)
    ax.plot(it, Y.mean(0), color=c, lw=2, label=label)


def two_panel(runs, title, fname, labels=("Standard SGD", "Eq. (1), FIM-scaled noise")):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for key, c, lab in zip(("sgd", "fim_noise"), (C_SGD, C_FIM), labels):
        it, L = curves(runs, key)
        band(a1, it, L, c, lab)
        band(a2, it, L, c, lab, log=True)
    a1.set(xlabel="iteration $n$", ylabel=r"full training-set loss $L(\theta_n)$")
    a2.set(xlabel="iteration $n$", ylabel=r"$\ln L(\theta_n)$")
    for a in (a1, a2):
        a.legend(frameon=False)
    fig.suptitle(title + "  (line: mean of 3 seeds; band: min-max over seeds)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(D, fname), dpi=200)
    plt.close(fig)


summary = {}

# ---------- E1: small init, SGD vs Eq. (1) ----------
e1 = load("E1_smallinit")
it, Ls = curves(e1, "sgd")
_, Lf = curves(e1, "fim_noise")
summary["E1"] = [dict(seed=s, sgd_final=float(Ls[k, -1]), sgd_max_dev_from_ln10=float(np.abs(Ls[k] - LN10).max()),
                      sgd_min_batch_loss=float(min(e1[k]["sgd"]["batch_loss"])),
                      fim_escape_iter=escape_iter(it, Lf[k]), fim_final=float(Lf[k, -1]))
                 for k, s in enumerate(SEEDS)]
two_panel(e1, r"E1: CIFAR-10, $\theta_0 = 0.1\times$ default init, $\eta=0.01$, $\rho_1=10^{-7}$, $\rho_2=10^{-6}$",
          "fig_E1_escape_smallinit.png")

# ---------- E2: default init ----------
e2 = load("E2_defaultinit")
it2, Ls2 = curves(e2, "sgd")
_, Lf2 = curves(e2, "fim_noise")
summary["E2"] = [dict(seed=s, sgd_final=float(Ls2[k, -1]), fim_final=float(Lf2[k, -1]),
                      max_abs_gap=float(np.abs(Ls2[k] - Lf2[k]).max()),
                      sgd_escape_iter=escape_iter(it2, Ls2[k]), fim_escape_iter=escape_iter(it2, Lf2[k]))
                 for k, s in enumerate(SEEDS)]
two_panel(e2, r"E2: CIFAR-10, default init, $\eta=0.01$, $\rho_1=10^{-9}$, $\rho_2=10^{-6}$", "fig_E2_defaultinit.png")

# ---------- E3: noise-to-step-size ratio sweep ----------
rho2, eta = 1e-6, 0.01
levels = [("1e-8", "E3_rho1_1e-8"), ("3e-8", "E3_rho1_3e-8"), ("1e-7", "E1_smallinit"), ("3e-7", "E3_rho1_3e-7")]
cols = plt.cm.Oranges(np.linspace(0.4, 0.95, len(levels)))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
it, L = curves(e1, "sgd")
a1.plot(it, L.mean(0), color=C_SGD, lw=2, label="Standard SGD")
summary["E3"] = []
for (r, name), c in zip(levels, cols):
    runs = load(name)
    it, L = curves(runs, "fim_noise")
    ratio = eta * (float(r) / rho2) ** 2
    lab = rf"$\rho_1$={r}: $\eta(\rho_1/\rho_2)^2$={ratio:.0e}"
    band(a1, it, L, c, lab)
    esc = [escape_iter(it, L[k]) for k in range(len(SEEDS))]
    summary["E3"].append(dict(rho1=float(r), rho1_over_rho2=float(r) / rho2, eta_ratio_sq=ratio, escape_iters=esc,
                              final=[float(x) for x in L[:, -1]]))
    xs = [ratio] * len(esc)
    ok = [e for e in esc if e is not None]
    a2.scatter([ratio] * len(ok), ok, color=c, s=40, zorder=3, edgecolor="white", lw=0.8)
    n_no = sum(e is None for e in esc)
    if n_no:
        a2.scatter([ratio], [it[-1]], marker="^", color=c, s=60, zorder=3, edgecolor="white", lw=0.8)
        a2.annotate(f"{n_no}/3 no escape\nin {it[-1]} it.", (ratio, it[-1]), textcoords="offset points",
                    xytext=(8, -18), fontsize=8)
a1.set(xlabel="iteration $n$", ylabel=r"full training-set loss $L(\theta_n)$")
a1.legend(frameon=False, fontsize=8)
a2.set(xscale="log", yscale="log", xlabel=r"noise-to-step-size ratio $\eta\,(\rho_1/\rho_2)^2$",
       ylabel="escape iteration (first $L<\\ln 10-0.1$)")
fig.suptitle(r"E3: escape vs. noise level, $\theta_0=0.1\times$ default init, $\eta=0.01$, $\rho_2=10^{-6}$ "
             "(left: mean, band min-max over 3 seeds; right: one dot per seed)", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "fig_E3_noise_ratio.png"), dpi=200)
plt.close(fig)

# ---------- E4: Robbins-Monro step vs constant step (Eq. 1) ----------
e4 = load("E4_rm")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
for runs, key, c, ls, lab in [(e1, "fim_noise", C_FIM, "-", r"Eq. (1), constant $\alpha_n=\eta$"),
                              (e4, "fim_noise", C_FIM, "--", r"Eq. (1), $\alpha_n=\eta\,n_0/(n_0+n)$"),
                              (e4, "sgd", C_SGD, "--", r"SGD, $\alpha_n=\eta\,n_0/(n_0+n)$")]:
    it, L = curves(runs, key)
    for a, Y in ((a1, L), (a2, np.log(L))):
        a.fill_between(it, Y.min(0), Y.max(0), color=c, alpha=0.15, lw=0)
        a.plot(it, Y.mean(0), color=c, ls=ls, lw=2, label=lab)
a1.axhline(ESC, color="0.5", lw=0.8, ls=":")
a1.text(it[-1], ESC, " plateau band", va="bottom", ha="right", fontsize=8, color="0.35")
a1.set(xlabel="iteration $n$", ylabel=r"full training-set loss $L(\theta_n)$")
a2.set(xlabel="iteration $n$", ylabel=r"$\ln L(\theta_n)$")
a1.legend(frameon=False, fontsize=8)
fig.suptitle(r"E4: no slide-back after escape; constant vs. Robbins-Monro step ($\eta=0.01$, $n_0=2000$), "
             r"$\theta_0=0.1\times$ default, $\rho_1=10^{-7}$ (mean, band min-max over 3 seeds)", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "fig_E4_no_slideback.png"), dpi=200)
plt.close(fig)
summary["E4"] = []
for name, runs in (("const", e1), ("rm", e4)):
    it, L = curves(runs, "fim_noise")
    _, Ls_ = curves(runs, "sgd")
    for k, s in enumerate(SEEDS):
        summary["E4"].append(dict(step=name, seed=s, escape_iter=escape_iter(it, L[k]), final=float(L[k, -1]),
                                  sgd_final=float(Ls_[k, -1]), **(slide_back(it, L[k]) or {})))

# ---------- E5: post-escape rate (log L, aligned at escape) ----------
fig, ax = plt.subplots(figsize=(6, 3.8))
summary["E5"] = []
for name, runs, ls in (("const", e1, "-"), ("rm", e4, "--")):
    it, L = curves(runs, "fim_noise")
    for k, s in enumerate(SEEDS):
        t = escape_iter(it, L[k])
        m = it >= t
        ax.plot(it[m] - t, np.log(L[k, m]), color=C_FIM, ls=ls, lw=1.2, alpha=0.5 + 0.25 * k,
                label=("constant step" if name == "const" else "Robbins-Monro step") if k == 0 else None)
        summary["E5"].append(dict(step=name, seed=s, escape_iter=t,
                                  slopes_lnL_per_1k=[dict(window=f"{a}-{b}", slope=sl) for a, b, sl in slopes(it, L[k], t)]))
ax.set(xlabel=r"iterations since escape, $n-\tau$", ylabel=r"$\ln L(\theta_n)$")
ax.legend(frameon=False)
ax.set_title(r"E5: post-escape $\ln L$, Eq. (1), 3 seeds per step rule ($L^*$ unknown, $L_{\mathrm{ref}}=0$)", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "fig_E5_post_escape_rate.png"), dpi=200)
plt.close(fig)

json.dump(summary, open(os.path.join(D, "summary_numbers.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
