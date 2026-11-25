"""Publication-quality figures for the two final experiments.
Reads the EXISTING raw results in place (nothing is re-run, nothing is modified):
  ../final_experiment_1_saddle_escape/raw/E1_smallinit_seed{0,1,2}.json
  ../final_experiment_2_handover_convergence/{summary.json, raw/*.npz}
Writes fig1..fig4 as .png (300 dpi) and .pdf into this folder.   Run: python3 make_clean_figures.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.join(HERE, "..", "final_experiment_1_saddle_escape", "raw")
E2 = os.path.join(HERE, "..", "final_experiment_2_handover_convergence")
ETA, N0 = 0.01, 1000  # must match run_experiment2.py
C = dict(sgd="#2a78d6", fim="#eb6834", iso="#1baf7a", off="#6250d6", region="#e34948", ink="#222222")
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.5, "legend.frameon": False, "lines.linewidth": 1.5,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


def letter(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, f"({s})", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def loss2(q, a, b):
    return 0.1 * ((a ** 4 - 1) ** q + b ** 2)


def iqr(ax, x, st, c, label, ls="-", lw=1.5, alpha=0.18):
    """Median line with interquartile band (st columns: 5,25,50,75,95 percentiles, mean)."""
    m = np.isfinite(st[:, 2])
    ax.fill_between(x[m], st[m, 1], st[m, 3], color=c, alpha=alpha, lw=0)
    ax.plot(x[m], st[m, 2], color=c, ls=ls, lw=lw, label=label)


# =============================================================================== Figure 1: CIFAR-10 escape
runs = [json.load(open(os.path.join(E1, f"E1_smallinit_seed{s}.json"))) for s in (0, 1, 2)]
it = np.array(runs[0]["sgd"]["eval_iters"]) / 1000
fig = plt.figure(figsize=(7.0, 6.4))
ax1, ax2 = fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2)
for key, c, lab in (("sgd", C["sgd"], "SGD"), ("fim_noise", C["fim"], "Eq. (1)")):
    L = np.array([r[key]["eval_loss"] for r in runs])
    for ax, Y in ((ax1, L), (ax2, np.log(L))):
        ax.fill_between(it, Y.min(0), Y.max(0), color=c, alpha=0.25, lw=0)
        ax.plot(it, Y.mean(0), color=c, label=lab)
ax1.set(xlabel=r"iteration $n$ ($\times 10^3$)", ylabel=r"training loss $L(\theta_n)$", ylim=(0.85, 2.42))
ax2.set(xlabel=r"iteration $n$ ($\times 10^3$)", ylabel=r"$\ln L(\theta_n)$")
ax1.annotate(r"$\ln 10$ (uniform prediction)", xy=(12, np.log(10)), xytext=(9.5, 2.0), fontsize=8, color="0.3",
             arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6))
ax1.axvline(1.5, color="0.4", ls=":", lw=0.8)
ax1.text(1.75, 1.0, "escape\n$n=1{,}500$\n(all seeds)", fontsize=7.5, color="0.3")
ax1.legend(loc="center right")
letter(ax1, "a"); letter(ax2, "b")
axis = np.array(runs[0]["axis"])
A, B = np.meshgrid(axis, axis)
cc = len(axis) // 2
CLIP = 10.0
for k, (key, title) in enumerate((("sgd", "around final SGD weights"), ("fim_noise", "around final Eq. (1) weights"))):
    Z = np.minimum(np.nan_to_num(np.array(runs[0][key]["landscape"], float), nan=CLIP, posinf=CLIP), CLIP)
    ax = fig.add_subplot(2, 2, 3 + k, projection="3d")
    ax.plot_surface(A, B, Z, cmap="viridis", vmin=0, vmax=CLIP, lw=0, antialiased=True, rstride=1, cstride=1, alpha=0.95)
    ax.scatter([0], [0], [Z[cc, cc]], color=C["region"], s=14, depthshade=False)
    ax.set_zlim(0, CLIP)
    ax.set_xlabel(r"$\alpha$", labelpad=-6); ax.set_ylabel(r"$\beta$", labelpad=-6); ax.set_zlabel("loss", labelpad=-8)
    ax.set_box_aspect((1, 1, 0.75), zoom=0.92)
    ax.tick_params(pad=-3, labelsize=7)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_zticks([0, 5, 10])
    ax.view_init(28, -55)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.0)
    ax.text2D(0.0, 1.0, f"({'cd'[k]})", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    ax.text2D(0.12, 1.0, f"{title},  " + rf"$L(\theta^*)={Z[cc, cc]:.3f}$", transform=ax.transAxes, fontsize=8.5, va="bottom")
fig.subplots_adjust(hspace=0.38, wspace=0.3)
save(fig, "fig1_cifar_saddle_escape")

# =============================================================================== Experiment 2 data
S = json.load(open(os.path.join(E2, "summary.json")))
ld = lambda n: np.load(os.path.join(E2, "raw", n + ".npz"))
TAG = {2: "PL ($q=2$)", 4: r"KL ($q=4$, $\kappa=3/4$)"}


def pick(d):
    """One trajectory that stays in its first basin and one that slides back (from the 20 stored)."""
    switched = d["basin_switches"][:20] > 0
    stay = int(np.flatnonzero(~switched)[0])
    other = switched & (np.sign(d["keep_a"][-1].astype(float)) != d["sign_at_handover"][:20])
    sw = np.flatnonzero(other) if other.any() else np.flatnonzero(switched)  # prefer one that ends in the other basin
    return stay, (int(sw[0]) if len(sw) else None)


# =============================================================================== Figure 2: geometry + trajectories
fig = plt.figure(figsize=(7.0, 6.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1])
for row, q in enumerate((2, 4)):
    T = S[f"q{q}"]["theory"]
    r, ac = T["r"], T["a_crit"]
    d = ld(f"q{q}_saddle_fim_const")
    G, KA, KB = d["grid"], d["keep_a"].astype(float), d["keep_b"].astype(float)
    stay, sw = pick(d)
    m = G <= 1500
    # 3D surface
    ax = fig.add_subplot(gs[row, 0], projection="3d")
    aa, bb = np.meshgrid(np.linspace(-1.2, 1.2, 121), np.linspace(-0.8, 0.8, 81))
    zc = 0.2  # L <= 0.2 on this whole domain for both q, so nothing is clipped
    ax.plot_surface(aa, bb, loss2(q, aa, bb), cmap="Greys", vmin=-0.1, vmax=0.3, lw=0, alpha=0.55, rstride=2, cstride=2)
    tm = m & (np.abs(KB[:, stay]) <= 0.8)
    ax.plot(KA[tm, stay], KB[tm, stay], loss2(q, KA[tm, stay], KB[tm, stay]) + 0.003, color=C["fim"], lw=0.8)
    for s_ in (-1, 1):
        yb = np.linspace(-0.8, 0.8, 40)
        ax.plot(np.full_like(yb, s_ * ac), yb, loss2(q, s_ * ac, yb) + 0.002, color=C["region"], lw=1.1)
    ax.scatter([0], [0], [0.1], color=C["sgd"], s=22, depthshade=False)
    ax.scatter([-1, 1], [0, 0], [0.0, 0.0], color=C["ink"], marker="*", s=40, depthshade=False)
    ax.set_xlabel("$a$", labelpad=-6); ax.set_ylabel("$b$", labelpad=-6)
    ax.tick_params(pad=-3, labelsize=7)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-0.5, 0, 0.5]); ax.set_zticks([0, 0.1, 0.2]); ax.set_zlim(0, zc)
    ax.set_box_aspect((1.4, 1, 0.75), zoom=1.15)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.0)
    ax.view_init(30, -60)
    ax.text2D(0.0, 0.97, f"({'ac'[row]})", transform=ax.transAxes, fontsize=10, fontweight="bold")
    ax.text2D(0.12, 0.97, TAG[q] + r":  surface $L(a,b)$", transform=ax.transAxes, fontsize=9)
    # contour
    ax = fig.add_subplot(gs[row, 1])
    aa, bb = np.meshgrid(np.linspace(-1.3, 1.3, 400), np.linspace(-1.0, 1.0, 300))
    ax.contourf(aa, bb, np.log10(loss2(q, aa, bb) + 1e-5), levels=14, cmap="Greys_r", alpha=0.35)
    for s_ in (-1, 1):
        ax.axvspan(s_ * ac, s_ * 1.3, color=C["region"], alpha=0.08, lw=0)
        ax.axvline(s_ * ac, color=C["region"], lw=1.0)
        ax.axvline(s_ * r, color=C["ink"], ls="--", lw=0.9)
    for j, c_, lw_ in ((stay, C["fim"], 0.8), (sw, C["iso"], 0.8)):
        if j is None:
            continue
        ax.plot(KA[m, j], KB[m, j], color=c_, lw=lw_, alpha=0.9)
        ie, ih = np.searchsorted(G, d["t_esc"][j]), np.searchsorted(G, d["t_handover"][j])
        ax.scatter(KA[ih, j], KB[ih, j], s=26, facecolor="white", edgecolor=C["region"], lw=1.1, zorder=5)
        ax.scatter(KA[ie, j], KB[ie, j], s=26, marker="x", color=C["ink"], lw=1.1, zorder=5)
    ax.scatter([0], [0], color=C["sgd"], s=22, zorder=6)
    ax.scatter([-1, 1], [0, 0], color=C["ink"], marker="*", s=45, zorder=6)
    ax.set(xlim=(-1.3, 1.3), ylim=(-1.0, 1.0), xlabel="$a$  (escape direction $u$)", ylabel="$b$")
    ax.text(0.5 * (ac + 1.3), 0.88, "certified\nregion $D$", ha="center", va="top", fontsize=7.5, color=C["region"])
    letter(ax, "bd"[row], x=-0.16)
handles = [Line2D([], [], color=C["fim"], lw=1.2, label="Eq. (1) trajectory that stays"),
           Line2D([], [], color=C["iso"], lw=1.2, label="Eq. (1) trajectory that slides back"),
           Line2D([], [], color=C["region"], lw=1.2, label=r"certified boundary $|a|=a_c$"),
           Line2D([], [], color=C["ink"], ls="--", lw=1.0, label=r"escape radius $|a|=r$ (Thm 4 / Cor. 1)"),
           Line2D([], [], marker="o", ls="", mfc="white", mec=C["region"], label="handover (first entry into $D$)"),
           Line2D([], [], marker="x", ls="", color=C["ink"], label=r"escape ($|a|\geq r$)"),
           Line2D([], [], marker="o", ls="", color=C["sgd"], label="saddle = start (SGD never moves)"),
           Line2D([], [], marker="*", ls="", color=C["ink"], ms=8, label="global minima $(\\pm 1,0)$")]
fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=7.5)
fig.subplots_adjust(hspace=0.28, wspace=0.32, bottom=0.12)
save(fig, "fig2_handover_geometry")

# =============================================================================== Figure 3: loss gap + certified ratio
fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.4), sharex=True)
for col, q in enumerate((2, 4)):
    T = S[f"q{q}"]["theory"]
    fc, ic, sc, off = (ld(f"q{q}_saddle_{x}") for x in ("fim_const", "iso_const", "sgd_const", "fim_const_noiseoff"))
    x = fc["grid"] + 1.0
    med_h, med_e = np.median(fc["t_handover"]), np.median(fc["t_esc"])
    ax = axs[0, col]
    iqr(ax, x, sc["stat_loss"], C["sgd"], "SGD")
    iqr(ax, x, ic["stat_loss"], C["iso"], "isotropic noise (control)")
    iqr(ax, x, fc["stat_loss"], C["fim"], "Eq. (1)")
    iqr(ax, x, off["stat_loss"], C["off"], "Eq. (1), noise off after handover", ls="--")
    ax.set(xscale="log", yscale="log", ylim=(1e-12, 0.5))
    ax.set_title(TAG[q], pad=4)
    for a_ in axs[:, col]:
        a_.axvline(med_h + 1, color=C["region"], lw=0.9)
        a_.axvline(med_e + 1, color=C["ink"], ls="--", lw=0.9)
    ax.text(med_h * 0.9, 0.35, r"$\tau_h$", ha="right", va="top", fontsize=8, color=C["region"])
    ax.text(med_e * 1.12, 0.35, r"$\tau_{esc}$", ha="left", va="top", fontsize=8)
    ax = axs[1, col]
    iqr(ax, x, fc["stat_ratio"], "0.55", "measured ratio $R(\\theta_n)$", lw=1.0, alpha=0.12)
    iqr(ax, x, fc["stat_ratio_lb"], C["fim"], r"certified lower bound $\psi(\theta_n)$")
    ax.axhline(T["mu"], color=C["ink"], lw=1.0, ls=":")
    ax.text(3e4, T["mu"] * 0.55, rf"$\mu={T['mu']:.3g}$", fontsize=8, ha="right", va="top")
    ax.set(xscale="log", yscale="log", ylim=(1e-5, 10), xlabel="iteration $n+1$")
    letter(axs[0, col], "ab"[col]); letter(axs[1, col], "cd"[col])
axs[0, 0].set_ylabel(r"loss gap $L(\theta_n)-L^*$")
axs[1, 0].set_ylabel(r"$R_{PL}=\|\nabla L\|^2/2(L-L^*)$")
axs[1, 1].set_ylabel(r"$R_{KL}=\|\nabla L\|/(L-L^*)^{3/4}$")
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03), fontsize=7.5)
axs[1, 0].legend(loc="lower right", fontsize=7)
fig.subplots_adjust(hspace=0.25, wspace=0.3, top=0.9)
save(fig, "fig3_escape_and_certified_handover")

# =============================================================================== Figure 4: retention + convergence
fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6))
# (a) retention distances, PL
q, T = 2, S["q2"]["theory"]
fc, ic = ld("q2_saddle_fim_const"), ld("q2_saddle_iso_const")
x = fc["grid"] + 1.0
ax = axs[0, 0]
iqr(ax, x, ic["stat_abs_a"], C["iso"], r"$|a_n|$, isotropic noise")
iqr(ax, x, fc["stat_abs_a"], C["fim"], r"$|a_n|$, Eq. (1)")
iqr(ax, x, fc["stat_dist_min"], C["off"], "distance to destination minimum, Eq. (1)", ls="--")
ax.axhline(T["a_crit"], color=C["region"], lw=1.0)
ax.axhline(T["r"], color=C["ink"], ls="--", lw=0.9)
ax.text(1.3, T["a_crit"] * 0.78, r"$a_c$", color=C["region"], fontsize=8)
ax.text(1.3, T["r"] * 1.08, r"$r$", fontsize=8)
ax.set(xscale="log", yscale="log", ylim=(3e-3, 3), xlabel="iteration $n+1$", ylabel="distance in parameter space",
       title=TAG[2])
ax.legend(loc="lower right", fontsize=7)
# (b) outcome bars
ax = axs[0, 1]
cfgs = [("fim_const", "Eq. (1)\nconst."), ("fim_rm", "Eq. (1)\nR–M"), ("iso_const", "isotropic\nconst."), ("iso_rm", "isotropic\nR–M")]
w = 0.19
for qi, q_ in enumerate((2, 4)):
    sw, st = [], []
    for nm, _ in cfgs:
        d_ = ld(f"q{q_}_saddle_{nm}")
        sw.append(S[f"q{q_}"]["runs"][f"q{q_}_saddle_{nm}"]["pct_basin_switch_after_handover"])
        st.append(100 * np.mean((d_["t_handover"] >= 0) & (d_["last_region_exit"] < 5000)))
    xs = np.arange(len(cfgs))
    hatch = "" if q_ == 2 else "////"
    ax.bar(xs - 1.5 * w + qi * w, sw, w * 0.92, color=C["region"], alpha=0.85, hatch=hatch, edgecolor="white", lw=0)
    ax.bar(xs + 0.5 * w + qi * w, st, w * 0.92, color=C["sgd"], alpha=0.85, hatch=hatch, edgecolor="white", lw=0)
    for xi, v in zip(xs + 0.5 * w + qi * w, st):
        if v < 1:
            ax.text(xi, 1.5, "0", ha="center", va="bottom", fontsize=7, color=C["sgd"])
ax.set_xticks(np.arange(len(cfgs)), [n for _, n in cfgs], fontsize=7.5)
ax.set(ylabel="% of 200 trajectories", ylim=(0, 108), title="retention after handover")
ax.grid(axis="x", visible=False)
ax.legend(handles=[Patch(color=C["region"], label="slid back through saddle"),
                   Patch(color=C["sgd"], label="no region exit after $n=5{,}000$"),
                   Patch(facecolor="0.6", label="PL"), Patch(facecolor="0.6", hatch="////", edgecolor="white", label="KL")],
          loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.0))
ax.set_ylim(0, 130); ax.set_yticks([0, 25, 50, 75, 100])
# (c) PL convergence after handover (semilog, iterations since handover)
ax = axs[1, 0]
fc, fr, off = ld("q2_saddle_fim_const"), ld("q2_saddle_fim_rm"), ld("q2_saddle_fim_const_noiseoff")
k = fc["post_grid"].astype(float)
iqr(ax, k, fc["post_stats"], C["fim"], "Eq. (1), const. step")
iqr(ax, k, fr["post_stats"], C["fim"], "Eq. (1), R–M step", ls="--")
iqr(ax, k, off["post_stats"], C["off"], r"noise off after $\tau_h$")
L0 = off["post_stats"][0, 2]
ax.plot(k, L0 * (1 - 0.2 * ETA) ** (2 * k), color=C["ink"], ls="-.", lw=0.9, label=r"$(1-\eta\lambda_{\min})^{2k}$")
ax.plot(k, L0 * (1 - ETA * T["mu"]) ** k, color=C["ink"], ls=":", lw=1.0, label=r"$(1-\eta\mu)^{k}$ (certified)")
ax.set(yscale="log", xlim=(0, 6000), ylim=(1e-13, 0.3), xlabel=r"iterations since handover $k=n-\tau_h$",
       ylabel=r"$L(\theta)-L^*$", title="PL: linear (geometric) convergence")
ax.legend(loc="lower left", fontsize=6.5, handlelength=1.8)
# (d) KL convergence (log-log vs accumulated step)
ax = axs[1, 1]
fc, fr, off, ic = (ld(f"q4_saddle_{x}") for x in ("fim_const", "fim_rm", "fim_const_noiseoff", "iso_const"))
k = fc["post_grid"].astype(float)


def acc(kk, t0):
    n = np.arange(int(t0), int(t0) + int(kk.max()) + 1)
    return np.concatenate([[0.0], np.cumsum(ETA * N0 / (N0 + n))])[kk.astype(int)]


iqr(ax, ETA * k + ETA, ic["post_stats"], C["iso"], "isotropic, constant step")
iqr(ax, ETA * k + ETA, fc["post_stats"], C["fim"], "Eq. (1), constant step")
iqr(ax, acc(k, np.median(fr["t_handover"])) + ETA, fr["post_stats"], C["fim"], "Eq. (1), Robbins–Monro step", ls="--")
iqr(ax, ETA * k + ETA, off["post_stats"], C["off"], r"noise off after $\tau_h$ (diagnostic)")
tt = np.geomspace(3, 500, 30)
ax.plot(tt, 1.2e-3 * (tt / 3) ** -2.0, color=C["ink"], ls=":", lw=1.0, label=r"KL prediction $\propto t^{-1/(2\kappa-1)}=t^{-2}$")
ax.set(xscale="log", yscale="log", xlim=(1e-2, 600), ylim=(1e-9, 0.5), xlabel=r"accumulated step $t=\sum\alpha_n$ since handover",
       ylabel=r"$L(\theta)-L^*$", title="KL: sublinear convergence; noise floor")
ax.legend(loc="lower left", fontsize=6.8)
for a_, s in zip(axs.flat, "abcd"):
    letter(a_, s)
fig.subplots_adjust(hspace=0.42, wspace=0.32)
save(fig, "fig4_retention_and_convergence")
print("written:", sorted(f for f in os.listdir(HERE) if f.startswith("fig")))
