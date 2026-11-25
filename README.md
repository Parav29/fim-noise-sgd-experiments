# FIM-scaled noise SGD — experiments

Experiments for *Polynomial-Time Handovers from Strict and Degenerate Saddles to Local Polyak–Łojasiewicz Basins in Deep Learning* (Ramaswamy, Solanki, Borkar).

| Folder | Content |
|---|---|
| `final_experiment_1_saddle_escape/` | CIFAR-10 CNN started near θ = 0: SGD stalls, Eq. (1) escapes (3 seeds) |
| `final_experiment_2_handover_convergence/` | 2-parameter deep linear net with a 4th-order saddle: escape → certified PL/KL region → retention → convergence (200 trajectories per configuration) |
| `final_experiments_clean_figures/` | Paper figures (PDF/PNG) and draft captions, made from the two folders above |
| `fim_noise_sgd.py` | Eq. (1) optimizer and CIFAR-10 training, used by Experiment 1 |

Each folder's `README.md` covers its setup, results and limitations. CIFAR-10 downloads to `data/` on first run. Earlier exploratory runs are in the git history (before commit `4af6a1f`).
