"""Regression gate: the generalised core at p=4, eta=0.01, r_frac=0.5, noise_mult=1 and the original seeds must
reproduce ../summary.json (theory constants to 1e-12 relative, every run-summary field exactly, all 28 runs).
Also runs the p-layer torch gradient check for every (p, q) used in the ablation.  Exit code 1 on any mismatch."""
import json, math, os, sys
import numpy as np
from ablation_core import theory, torch_check, simulate, summarize, INITS

HERE = os.path.dirname(os.path.abspath(__file__))
ref = json.load(open(os.path.join(HERE, "..", "summary.json")))
bad = []

for p in (2, 4, 6, 8):
    for q in (2, 4):
        torch_check(p, q)
print("torch p-layer gradient check: ok for p in {2,4,6,8}, q in {2,4}")

for q in (2, 4):
    th = theory(4, q)
    for k, v in ref[f"q{q}"]["theory"].items():
        if isinstance(v, bool) or k not in th or isinstance(v, str):
            continue
        w = th[k]
        same = (v == w) if (isinstance(v, int) or v is None or (isinstance(v, float) and math.isinf(v))) \
            else abs(w - v) <= 1e-12 * max(1.0, abs(v))
        if not same:
            bad.append(f"theory q{q}.{k}: ref {v} new {w}")
    for ii, (iname, init) in enumerate(INITS.items()):
        for opt in ("sgd", "fim", "iso"):
            for step in ("const", "rm"):
                for sched in (("const", "off") if (opt, step) == ("fim", "const") else ("const",)):
                    name = f"q{q}_{iname}_{opt}_{step}" + ("_noiseoff" if sched == "off" else "")
                    common, (ev,) = simulate(4, q, th, init, {"sgd": "none"}.get(opt, opt), step, sched, seed=100 * q + ii)
                    new = summarize(common, ev, th, 50_000)
                    old = ref[f"q{q}"]["runs"][name]
                    for k, v in old.items():
                        if json.dumps(v) != json.dumps(new[k]):
                            bad.append(f"{name}.{k}: ref {v} new {new[k]}")
                    print(f"  {name:30s} {'ok' if not any(b.startswith(name) for b in bad) else 'MISMATCH'}", flush=True)

print("\n".join(bad) if bad else "REGRESSION PASSED: all theory constants and all 28 run summaries match ../summary.json")
sys.exit(1 if bad else 0)
