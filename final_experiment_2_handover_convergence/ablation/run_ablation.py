"""Runs sweeps S1-S4 in parallel, resumable (a configuration is skipped if results/<name>.json exists).
  python3 run_ablation.py           full study (requires check_regression.py to have passed)
  python3 run_ablation.py --pilot   short pilot into pilot/ (stability and timing)
Seeds: 100*q + init_index for every configuration (common random numbers, same seeds as Experiment 2)."""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PS, QS = (2, 4, 6, 8), (2, 4)
S1_OPTS = {"sgd_const": ("none", "const", "const"), "fim_const": ("fim", "const", "const"),
           "iso_const": ("iso", "const", "const"), "sign_const": ("sign", "const", "const"),
           "fim_rm": ("fim", "rm", "const"), "iso_rm": ("iso", "rm", "const"),
           "fim_off": ("fim", "const", "off"), "fim_decay": ("fim", "const", "decay")}
NOISES = ("fim", "iso", "sign")


def tasks():
    out = []
    for p in PS:
        for q in QS:
            for ii, init in enumerate(("saddle", "near")):
                for lab, (noise, step, sched) in S1_OPTS.items():
                    out.append(dict(name=f"S1_p{p}_q{q}_{init}_{lab}", sweep="S1", p=p, q=q, init=init, ii=ii,
                                    noise=noise, step=step, sched=sched, eta=0.01, mult=1.0, N=50_000, full=True))
            for noise in NOISES:
                base = dict(p=p, q=q, init="saddle", ii=0, noise=noise, step="const", sched="const", full=False)
                for eta in (0.02, 0.005, 0.0025):
                    for var in ("thm", "fixednoise", "efftime"):
                        # fixednoise: rho1 kept at its eta=0.01 value (rho1 ~ mult/sqrt(eta)); efftime: N*eta = 500
                        out.append(dict(base, name=f"S2_p{p}_q{q}_eta{eta}_{var}_{noise}", sweep="S2", eta=eta, variant=var,
                                        mult=(np.sqrt(eta / 0.01) if var == "fixednoise" else 1.0),
                                        N=(int(round(50_000 * 0.01 / eta)) if var == "efftime" else 50_000)))
                for mult in (0.25, 0.5, 2.0, 4.0):
                    out.append(dict(base, name=f"S3_p{p}_q{q}_mult{mult}_{noise}", sweep="S3", eta=0.01, mult=mult, N=50_000))
                out.append(dict(base, name=f"S4_p{p}_q{q}_{noise}", sweep="S4", eta=0.01, mult=1.0, N=50_000,
                                r_fracs=(0.5, 0.25, 0.75, 1.0)))   # one simulation, four event definitions
    return out


def run(t, res_dir, raw_dir):
    from ablation_core import theory, simulate, summarize, INITS
    names = [f"{t['name']}_r{rf}" for rf in t["r_fracs"]] if "r_fracs" in t else [t["name"]]
    if all(os.path.exists(os.path.join(res_dir, n + ".json")) for n in names):
        return f"skip {t['name']}"
    t0 = time.time()
    ths = [theory(t["p"], t["q"], t["eta"], rf, t["mult"]) for rf in t.get("r_fracs", (0.5,))]
    assert len({th["rho1"] for th in ths}) == 1, "radius sweep must not change the dynamics"
    common, evs = simulate(t["p"], t["q"], ths[0], INITS[t["init"]], t["noise"], t["step"], t["sched"],
                           seed=100 * t["q"] + t["ii"], N=t["N"], specs=ths, full=t["full"])
    arrays = dict(common)
    for k, ev in enumerate(evs):
        arrays.update({f"e{k}_{key}": v for key, v in ev.items()})
    np.savez_compressed(os.path.join(raw_dir, t["name"] + ".npz"), **arrays)
    cfg = {k: v for k, v in t.items() if k != "r_fracs"}
    for k, (n, th, ev) in enumerate(zip(names, ths, evs)):
        rec = dict(name=n, config=dict(cfg, r_frac=th["r_frac"], raw_file=f"raw/{t['name']}.npz", event_prefix=f"e{k}_"),
                   theory=th, summary=summarize(common, ev, th, t["N"], seed=100 * t["q"] + t["ii"]))
        json.dump(rec, open(os.path.join(res_dir, n + ".json"), "w"), indent=1, default=float)
    s = json.load(open(os.path.join(res_dir, names[0] + ".json")))["summary"]
    return (f"{t['name']:40s} {time.time() - t0:6.1f}s esc {s['pct_escaped']:5.1f}% reentry {s['pct_saddle_reentry']:5.1f}% "
            f"switch {s['pct_basin_switch_after_handover']:5.1f}% inD_end {s['pct_in_region_at_end']:5.1f}% div {s['pct_diverged']:5.1f}%")


def main():
    pilot = "--pilot" in sys.argv
    res_dir = os.path.join(HERE, "pilot" if pilot else "results")
    raw_dir = os.path.join(HERE, "pilot" if pilot else "raw")
    os.makedirs(res_dir, exist_ok=True); os.makedirs(raw_dir, exist_ok=True)
    ts = tasks()
    if pilot:  # extremes of every axis, short budget
        keep = lambda t: t["p"] in (2, 8) and t["init"] == "saddle" and (
            t["sweep"] == "S1" and t.get("noise") in ("fim", "iso") and t["step"] == "const" and t["sched"] == "const"
            or t["sweep"] == "S2" and t["eta"] == 0.02 and t["variant"] == "thm" and t["noise"] == "fim"
            or t["sweep"] == "S3" and t["mult"] == 4.0 and t["noise"] == "fim")
        ts = [dict(t, N=5000, name="pilot_" + t["name"]) for t in ts if keep(t)]
    print(f"{len(ts)} simulations -> {res_dir}", flush=True)
    log = open(os.path.join(HERE, "pilot_log.txt" if pilot else "run_log.txt"), "a")
    t0 = time.time()
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        for i, line in enumerate(pool.imap_unordered(lambda_run, [(t, res_dir, raw_dir) for t in ts])):
            msg = f"[{i + 1}/{len(ts)} {time.time() - t0:7.0f}s] {line}"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()


def lambda_run(args):
    return run(*args)


if __name__ == "__main__":
    main()
