"""SGD vs. FIM-scaled-noise SGD (Eq. 1, Ramaswamy & Borkar) on CIFAR-10.

theta_{n+1} = theta_n - eta * (grad L_n + Upsilon_n),
Upsilon_n^i = s_i * rho1 / (rho2 + (dL_n/dtheta_i)^2),  s_i = +-1 w.p. 1/2.
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class FIMNoiseSGD(torch.optim.Optimizer):
    """Eq. (1) of the paper, constant step size alpha_n = lr."""

    def __init__(self, params, lr, rho1, rho2, generator=None):
        super().__init__(params, dict(lr=lr, rho1=rho1, rho2=rho2))
        self.generator = generator

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, rho1, rho2 = group["lr"], group["rho1"], group["rho2"]
            for p in group["params"]:
                g = p.grad
                # s_i in {-1,+1} with prob 1/2, independent per coordinate and per step
                s = torch.randint(0, 2, g.shape, generator=self.generator, device=g.device, dtype=g.dtype) * 2 - 1
                p.sub_(lr * (g + s * rho1 / (rho2 + g * g)))


def selfcheck():
    p = nn.Parameter(torch.zeros(1000))
    g = torch.linspace(-0.1, 0.1, 1000)
    p.grad = g.clone()
    FIMNoiseSGD([p], lr=0.5, rho1=1e-3, rho2=1e-2).step()
    noise = -p.detach() / 0.5 - g
    assert torch.allclose(noise.abs(), 1e-3 / (1e-2 + g * g), rtol=1e-5)
    assert 0.4 < (noise > 0).float().mean() < 0.6


class SmallCNN(nn.Module):
    # no BatchNorm / Dropout on purpose: they change the dynamics being studied
    def __init__(self, pool="max"):
        super().__init__()
        # avg pooling keeps the net C^inf (paper Assumption 1); max pooling is not smooth
        self.pool = F.avg_pool2d if pool == "avg" else F.max_pool2d
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x):
        x = self.pool(F.gelu(self.c1(x)), 2)
        x = self.pool(F.gelu(self.c2(x)), 2)
        x = self.pool(F.gelu(self.c3(x)), 2)
        return self.fc(x.flatten(1))


def load_cifar(root, device):
    ds = torchvision.datasets.CIFAR10(root, train=True, download=True)
    x = torch.tensor(ds.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    return ((x - mean) / std).to(device), torch.tensor(ds.targets).to(device)


@torch.no_grad()
def full_train_loss(model, X, Y, bs=5000):
    return sum(F.cross_entropy(model(X[i:i + bs]), Y[i:i + bs], reduction="sum").item()
               for i in range(0, len(X), bs)) / len(X)


def random_directions(model, seed):
    """Two random Gaussian directions, filter-normalised (Li et al. 2018); biases get direction 0."""
    g = torch.Generator().manual_seed(seed)
    dirs = []
    for _ in range(2):
        ds = []
        for p in model.parameters():
            if p.dim() == 1:
                ds.append(torch.zeros_like(p)); continue
            d = torch.randn(p.shape, generator=g).to(p.device)
            ds.append(d * (p.flatten(1).norm(dim=1) / d.flatten(1).norm(dim=1)).view(-1, *[1] * (p.dim() - 1)))
        dirs.append(ds)
    return dirs


@torch.no_grad()
def landscape(model, Xe, Ye, axis, seed):
    """Loss at theta* + alpha*d1 + beta*d2 on the grid axis x axis (rows: beta, cols: alpha)."""
    d1, d2 = random_directions(model, seed)
    theta = [p.clone() for p in model.parameters()]
    Z = np.zeros((len(axis), len(axis)))
    for j, b in enumerate(axis):
        for i, a in enumerate(axis):
            for p, t, u, v in zip(model.parameters(), theta, d1, d2):
                p.copy_(t + a * u + b * v)
            Z[j, i] = full_train_loss(model, Xe, Ye)
    for p, t in zip(model.parameters(), theta):
        p.copy_(t)
    return Z


def train(method, init_state, X, Y, batches, args, device):
    model = SmallCNN(args.pool).to(device)
    model.load_state_dict(init_state)
    if method == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        gen = torch.Generator(device=device).manual_seed(args.seed + 1)
        opt = FIMNoiseSGD(model.parameters(), lr=args.lr, rho1=args.rho1, rho2=args.rho2, generator=gen)
    batch_loss, eval_iters, eval_loss = [], [], []
    t0 = time.time()
    for n, idx in enumerate(batches):
        if n % args.eval_every == 0:
            eval_iters.append(n); eval_loss.append(full_train_loss(model, X, Y))
            print(f"[{method}] iter {n:6d}  full-train loss {eval_loss[-1]:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if args.step == "rm":  # Robbins-Monro: alpha_n = lr*n0/(n0+n), sum alpha = inf, sum alpha^2 < inf (paper Thm 6)
            for g in opt.param_groups:
                g["lr"] = args.lr * args.rm_n0 / (args.rm_n0 + n)
        loss = F.cross_entropy(model(X[idx]), Y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        batch_loss.append(loss.item())
    eval_iters.append(len(batches)); eval_loss.append(full_train_loss(model, X, Y))
    print(f"[{method}] final full-train loss {eval_loss[-1]:.4f}", flush=True)
    return model, dict(batch_loss=batch_loss, eval_iters=eval_iters, eval_loss=eval_loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--rho1", type=float, default=1e-9)
    ap.add_argument("--rho2", type=float, default=1e-6)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--grid_n", type=int, default=31)  # landscape grid: alpha, beta in linspace(-1, 1, grid_n)
    ap.add_argument("--n_eval", type=int, default=5000)  # fixed training subset used for the landscape
    ap.add_argument("--init_scale", type=float, default=1.0)  # theta_0 = init_scale * default init; <1 starts near the saddle at 0
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", choices=["max", "avg"], default="max")
    ap.add_argument("--step", choices=["const", "rm"], default="const")
    ap.add_argument("--rm_n0", type=float, default=2000.0)
    ap.add_argument("--methods", default="sgd,fim_noise")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    selfcheck()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    X, Y = load_cifar(args.data, device)

    torch.manual_seed(args.seed)
    init_state = {k: args.init_scale * v for k, v in SmallCNN().state_dict().items()}  # same theta_0 for both runs
    rng = np.random.default_rng(args.seed)  # same mini-batch sequence for both runs
    nb = len(X) // args.batch_size  # drop last partial batch
    batches = [torch.from_numpy(perm[i * args.batch_size:(i + 1) * args.batch_size]).to(device)
               for _ in range(args.epochs) for perm in [rng.permutation(len(X))] for i in range(nb)]
    ev = torch.from_numpy(np.random.default_rng(args.seed).choice(len(X), args.n_eval, replace=False)).to(device)
    axis = np.linspace(-1, 1, args.grid_n) if args.grid_n > 1 else np.zeros(1)  # grid_n=1: theta* only

    res = dict(args=vars(args), device=device, axis=axis.tolist())
    for method in args.methods.split(","):
        model, res[method] = train(method, init_state, X, Y, batches, args, device)
        t0 = time.time()
        Z = landscape(model, X[ev], Y[ev], axis, args.seed + 1000)  # same raw directions for both runs
        c = args.grid_n // 2  # axis[c] == 0, i.e. theta* itself; also checks theta* was restored
        assert abs(Z[c, c] - full_train_loss(model, X[ev], Y[ev])) < 1e-4
        res[method]["landscape"] = Z.tolist()
        print(f"[{method}] landscape done ({time.time() - t0:.0f}s), min {Z.min():.4f}, max {Z.max():.4f}", flush=True)
    json.dump(res, open(args.out, "w"))


if __name__ == "__main__":
    main()
