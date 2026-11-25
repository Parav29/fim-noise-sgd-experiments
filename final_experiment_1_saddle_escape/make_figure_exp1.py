"""Final Experiment 1 figure: (1) training loss, (2) log training loss, (3)/(4) 3D loss landscapes around the final
SGD / FIM-noise SGD weights (seed 0). Reads raw/E1_smallinit_seed{0,1,2}.json written by run_exp1.sh."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
runs = [json.load(open(os.path.join(HERE, "raw", f"E1_smallinit_seed{s}.json"))) for s in (0, 1, 2)]
C = {"sgd": "#2a78d6", "fim_noise": "#eb6834"}
LAB = {"sgd": "Standard SGD", "fim_noise": "FIM-noise SGD, Eq. (1)"}
CLIP = 10.0
ESC = np.log(10) - 0.1
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})

it = np.array(runs[0]["sgd"]["eval_iters"])
summary = {}
fig = plt.figure(figsize=(13, 10))
a1, a2 = fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2)
for key in ("sgd", "fim_noise"):
    L = np.array([r[key]["eval_loss"] for r in runs])
    for ax, Y in ((a1, L), (a2, np.log(L))):
        ax.fill_between(it, Y.min(0), Y.max(0), color=C[key], alpha=0.2, lw=0)
        ax.plot(it, Y.mean(0), color=C[key], lw=2, label=LAB[key])
    esc = [int(it[np.argmax(l < ESC)]) if (l < ESC).any() else None for l in L]
    summary[key] = dict(final_loss_per_seed=L[:, -1].tolist(), final_loss_mean=float(L[:, -1].mean()),
                        max_abs_dev_from_ln10=float(np.abs(L - np.log(10)).max()), escape_iter_per_seed=esc)
for ax in (a1, a2):
    ax.axhline(np.log(10) if ax is a1 else np.log(np.log(10)), color="0.5", lw=0.8, ls=":")
    ax.set_xlabel("iteration $n$")
    ax.legend(loc="center right")
a1.set(ylabel=r"full training-set loss $L(\theta_n)$", title="(1) Training loss (mean of 3 seeds, band = min–max)")
a2.set(ylabel=r"$\ln L(\theta_n)$", title=r"(2) Log training loss ($L^*$ unknown, so this is not a gap)")
a1.text(it[-1] * 0.5, np.log(10) - 0.02, r"SGD stays at $\ln 10$ = 2.3026 (uniform guess)", ha="center", va="top", fontsize=8, color="0.35")

axis = np.array(runs[0]["axis"])
A, B = np.meshgrid(axis, axis)
c = len(axis) // 2
for k, key in enumerate(("sgd", "fim_noise")):
    Z = np.array(runs[0][key]["landscape"], dtype=float)
    Zc = np.minimum(np.nan_to_num(Z, nan=CLIP, posinf=CLIP), CLIP)
    ax = fig.add_subplot(2, 2, 3 + k, projection="3d")
    ax.plot_surface(A, B, Zc, cmap="viridis", vmin=0, vmax=CLIP, lw=0, alpha=0.9)
    ax.contour(A, B, Zc, levels=np.arange(0.25, CLIP, 0.5), cmap="viridis", offset=0, linewidths=0.6)
    ax.scatter([0], [0], [Zc[c, c]], color="#e34948", s=30, depthshade=False)
    ax.set_zlim(0, CLIP)
    ax.set(xlabel=r"$\alpha$", ylabel=r"$\beta$", zlabel="loss")
    ax.set_title(f"({3 + k}) Landscape around final {LAB[key]} weights, seed 0\n"
                 rf"$L(\theta^*+\alpha d_1+\beta d_2)$; $L(\theta^*)$ = {Zc[c, c]:.3f} (red dot); "
                 f"min {Z.min():.3f}, max {Z.max():.1f} (shown clipped at {CLIP:g})", fontsize=8.5)
    summary[key].update(landscape_center=float(Z[c, c]), landscape_min=float(Z.min()), landscape_max=float(Z.max()))
a = runs[0]["args"]
fig.suptitle(f"Final Experiment 1 — CIFAR-10, 66,570-parameter GeLU CNN (3 conv + avg-pool + linear), cross-entropy\n"
             f"θ₀ = {a['init_scale']}× default init, η = {a['lr']}, ρ₁ = {a['rho1']:g}, ρ₂ = {a['rho2']:g}, batch {a['batch_size']}, "
             f"{a['epochs']} epochs, 3 seeds", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_exp1_saddle_escape.png"), dpi=200)
json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
