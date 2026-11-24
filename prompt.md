# Task: Experimental Validation of FIM-Scaled Noise SGD

We are working on the paper **"Polynomial-Time Handovers from Strict and Degenerate Saddles to Local Polyak–Łojasiewicz Basins in Deep Learning"** (Ramaswamy & Borkar).

The professor has asked us to implement the paper's optimization algorithm and generate convergence plots that show how it behaves.

**Keep the experiment simple and focused. Do not go beyond the tasks below.**

---

## 1. Experimental Setup

Use Python and PyTorch.

- **Dataset:** CIFAR-10. Use the standard training set via `torchvision.datasets.CIFAR10`, normalized per channel, with no data augmentation. Report the training loss on the training set.
- **Model:** A small feedforward CNN with GeLU activations. For example, 2–3 conv layers followed by a linear classifier, with no batch normalization and no dropout. Both of those would change the optimization dynamics we are studying.
- **Loss function:** Cross-entropy
- **Training:** Both experiments use the same architecture, initial weights, dataset, mini-batch sequence, and learning rate.

---

## 2. Implement Two Optimization Methods

### Experiment A — Standard Gradient Descent

$$
\theta_{n+1} = \theta_n - \eta \nabla L_n
$$

### Experiment B — Proposed FIM-Scaled Noise Gradient Descent

Implement Equation (1) from Section 2 of the paper:

$$
\theta_{n+1} = \theta_n - \eta \left( \nabla L_n + \Upsilon_n \right)
$$

Each noise component is:

$$
\Upsilon_n^i = s_i \, \frac{\rho_1}{\rho_2 + \left( \dfrac{\partial L_n}{\partial \theta_i} \right)^2}
$$

- $s_i$ is randomly +1 or −1 with equal probability, drawn independently for each parameter at each step.
- $\rho_1, \rho_2 > 0$ are hyperparameters.
- $\partial L_n / \partial \theta_i$ is the same mini-batch gradient used in that step.

**Important:** Implement the equation exactly as written in the paper. Do not use Gaussian noise, and do not substitute Adam or any other optimizer.

---

## 3. Train and Compare

Train the same model separately with each optimization method.

- Record the training loss at every iteration.
- Keep the training configuration identical across both runs.
- Use fixed random seeds for reproducibility.
- Choose reasonable hyperparameters and document them.

The goal is to investigate whether the proposed noise helps the optimizer escape slow or flat regions of the loss and then converge faster.

---

## 4. Generate Two Plots

Put both plots in **one figure**.

### Plot 1 — Training Loss vs. Iterations

- Show both optimizers on the same graph.
- Look at whether standard gradient descent makes slow progress in some phase, and whether the noise-modulated version moves through that phase.

### Plot 2 — Logarithmic Loss Gap vs. Iterations

Plot:

$$
\log \left( L(\theta_n) - L_{\text{ref}} \right)
$$

- $L_{\text{ref}}$ is a reference minimum loss. State clearly how you defined it.
- This plot is for investigating the sublinear-to-linear convergence behavior discussed in the paper.

**Do not force the results to match the theory. Report what actually happens.**

---

## 5. Final Deliverables

Provide only:

1. A working PyTorch implementation of both optimizers.
2. The training experiment on CIFAR-10.
3. One figure containing the two comparison plots.
4. A brief summary of the results and the hyperparameters used.

---

## Scope Restriction

Do **not** implement any of the following:

- additional optimizers
- toy loss landscapes
- Hessian analysis
- extensive parameter sweeps
- extra datasets
- unrelated experiments

Focus on implementing Equation (1), comparing it with standard gradient descent, and generating the two plots the professor asked for.

If the results do not clearly show the expected transition, explain that limitation in the summary. Do not add new experiments on your own.