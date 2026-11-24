"""One figure: training loss vs. iteration, log(L - L_ref) vs. iteration, and the 2D loss landscape
L(theta* + alpha*d1 + beta*d2) around the final weights of each optimizer."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "results.json"
res = json.load(open(path))
a = res["args"]
L_REF = 0.0  # infimum of cross-entropy (attained only if the net interpolates the training set)
LOSS_CLIP = 10.0  # landscape values are clipped here for plotting (as in loss_landscape_pl_region)
runs = [("sgd", "Standard SGD (Exp. A)", "tab:blue"),
        ("fim_noise", rf"FIM-noise SGD, Eq. (1) (Exp. B; $\rho_1$={a['rho1']:g}, $\rho_2$={a['rho2']:g})", "tab:red")]

fig = plt.figure(figsize=(14, 11))
ax1, ax2 = fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2)
for key, label, c in runs:
    r = res[key]
    ax1.plot(r["batch_loss"], color=c, alpha=0.25, lw=0.5)
    ax1.plot(r["eval_iters"], r["eval_loss"], color=c, lw=2, label=label)
    ax2.plot(r["eval_iters"], np.log(np.array(r["eval_loss"]) - L_REF), color=c, lw=2, label=label)
ax1.set(xlabel="iteration n", ylabel="training cross-entropy",
        title="Training loss (thin: mini-batch loss each iteration; bold: full training-set loss)")
ax2.set(xlabel="iteration n", ylabel=r"$\log(L(\theta_n) - L_{\mathrm{ref}})$",
        title=r"Log loss gap, $L_{\mathrm{ref}}=0$, $L$ = full training-set loss")
for ax in (ax1, ax2):
    ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.title.set_fontsize(9)

A, B = np.meshgrid(res["axis"], res["axis"])
for k, (key, label, _) in enumerate(runs):
    Z = np.array(res[key]["landscape"], dtype=float)
    Zc = np.minimum(np.nan_to_num(Z, nan=LOSS_CLIP, posinf=LOSS_CLIP), LOSS_CLIP)
    ax = fig.add_subplot(2, 2, 3 + k, projection="3d")
    ax.plot_surface(A, B, Zc, cmap="viridis", vmin=0, vmax=LOSS_CLIP, linewidth=0, alpha=0.9)
    ax.contour(A, B, Zc, levels=np.arange(0.1, LOSS_CLIP, 0.5), cmap="viridis", offset=0, linewidths=0.6)
    c = len(res["axis"]) // 2
    ax.scatter([0], [0], [Zc[c, c]], color="red", s=30, depthshade=False)
    ax.set_zlim(0, LOSS_CLIP)
    ax.set(xlabel=r"$\alpha$", ylabel=r"$\beta$", zlabel="loss")
    ax.set_title(f"Loss landscape around final weights: {label.split(' (Exp')[0]}\n"
                 rf"$L(\theta^*+\alpha d_1+\beta d_2)$, filter-normalised random $d_1,d_2$; "
                 rf"$L(\theta^*)$={Zc[c, c]:.3f} (red dot)", fontsize=9)

fig.suptitle(f"CIFAR-10, small GeLU CNN (3 conv layers, no BN), lr={a['lr']}, batch={a['batch_size']}, "
             f"{a['epochs']} epochs, seed={a['seed']}, init scale={a.get('init_scale', 1.0):g}, {a.get('pool', 'max')} pool; landscapes on {a['n_eval']} fixed train images, clipped at {LOSS_CLIP:g}", fontsize=10)
fig.tight_layout()
out = path.replace(".json", "") + "_convergence_landscape.png"
fig.savefig(out, dpi=150)
print("saved", out)
