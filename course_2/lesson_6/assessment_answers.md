# Lesson 6 — Assessment (model answers)

These are concise reference answers aligned with Lesson 6. Your own write-up should still use complete sentences suitable for grading or peer review.

---

## Scenario 1: The architectural fork

### 1. Which path must you choose, and why?

**Choose the Primal Path** (solve directly in weights $\mathbf{w}$ and bias $b$, as in **`LinearSVC`**-style training). Fifteen million rows is far too large for the standard **dual / kernel** training picture, whose memory bottleneck is fundamentally tied to comparing **pairs** of training points at scale—not just “more CPU cycles.” You need algorithms whose dominant cost behaves like processing **many rows × feature dimension**, not allocating an **$N \times N$** similarity structure.

---

### 2. What mathematical limitation hits the kernel trick because of that choice?

On the **Primal Path** you do **not** get the textbook **dual kernel trick**: you are not optimizing the multipliers $\alpha_i$ on an objective where only **kernels** $K(\mathbf{x}_i,\mathbf{x}_j)$ appear. So you **cannot** plug in arbitrary $K(\cdot,\cdot)$ the way **`SVC`** does and expect the same scalability story.

If you tried the obvious **dual** route instead to regain kernels, fifteen million rows would force something on the order of an **$N \times N$ Gram matrix** of kernel evaluations for naive formulations—RAM dies before philosophy starts. **So at this scale the dual+/kernel combo is operationally ruled out.**

---

### 3. Two-step pipeline for non-linear boundaries at 15 million rows

**Approximate kernel space in finite dimensions**, then solve a **fast linear primal SVM**:

1. **Mapper (kernel approximation)** — e.g. **`RBFSampler`** from `sklearn.kernel_approximation`, which builds a randomized Fourier feature map. You transform each transaction into **`D`** explicit features (lesson example: **$D \approx 5{,}000$**), embedding “RBF-like” geometry without constructing an **$N \times N$** kernel matrix.

2. **Primal solver** — feed that **`N \times D$** matrix into **`LinearSVC`** (hinge-loss linear SVM, **`dual=False`** in sklearn is appropriate for fat feature matrices). Optimization runs in **mapped space**; nonlinear-looking boundaries appear when you interpret decisions back on the raw transaction axes.

Typical skeleton:

```python
Pipeline([
    ("fourier_mapper", RBFSampler(gamma=..., n_components=5000)),
    ("primal_solver", LinearSVC(loss="hinge", dual=False, max_iter=...)),
])
```

You are **explicitly constructing** approximate kernel coordinates, then exploiting **linear SVM primal speed**, instead of insisting on dual **implicit** kernels.

---

## Scenario 2: The mathematics of sparsity

**Start — hinge loss.** The separable-soft-margin dual is tied to the same feasibility story as **hinge** errors in the primal view. Margin-based training wants **near-zero sensitivity** when a point already sits comfortably on the correct side, beyond margin.

For a standard hinge-style score, per point,

$$
\ell_i \;=\; \max\bigl(0,\; 1 - y_i(\mathbf{w}\cdot \mathbf{x}_i + b)\bigr)\,.
$$

Whenever the model gives **distance beyond the slab** correctly—i.e., $y_i(\mathbf{w}\cdot \mathbf{x}_i + b) \geqslant 1$—the clamp $\max(0,\cdot)$ makes that point’s hinge contribution **exactly zero**. Loosely “the easy predictions don’t pay any tax”; they exert **no** tension on tightening the corridor.

---

**Middle — \(\alpha\) multipliers.** In the dual, **KKT** structure links each training constraint to one multiplier \(\alpha_i\). For inactive constraints (safely classified with margin slack to spare—where hinge would contribute nothing in primal language), complementary slackness drives **$\alpha_i = 0$**. Heavy lifters—“support vectors”—are exactly points with **$\alpha_i > 0$**.

---

**Production — inference and why the pickle shrinks.** Prediction with a trained kernel dual SVM is

$$
f(\mathbf{x}_{\mathrm{new}}) \;=\; \mathrm{sign}\!\left(\sum_i \alpha_i\, y_i\, K(\mathbf{x}_i,\mathbf{x}_{\mathrm{new}}) + b\right).
$$

Terms with \(\alpha_i=0\) are **literal zeros**. **Scikit-learn** therefore **drops** rows that contribute nothing—all non-support vectors—from the persisted model. Serialization stores **support vectors**, their labels, \(\alpha_i>0\), \(b\), and kernel hyperparameters—not the giant training CSV. Same idea for linear models after kernel approximation except the expanded features were materialized transiently during training unless you pickle the **`Pipeline`** (mapper + coefs).

Hence **800 MB \(\to\) ~1–2 MB** is normal: gigabytes were **inactive training rows**; inference never needed them once \(\alpha\) sparsifies the dual sum.

---

## Scenario 3: Variance and the box constraint

### 1. Which hyperparameter and which direction?

**Lower \(C\)**, the trade-off knob between **widening \(\lVert\mathbf{w}\rVert\)** versus **hammering hinge/slack**.

---

### 2. Box constraint and how \(C\) defangs outliers

**Box constraint:** in the dual (soft-margin SVM),

$$
0 \leqslant \alpha_i \leqslant C \, .
$$

Rough “pull” metaphor: $\alpha_i$ measures how loudly point $i$ is allowed to **force** boundary geometry at the optimum. **`C`** is the ceiling on that influence per point—a **budget cap.** Outliers begging for microscopic margins need **large** $\alpha$, but capped at **`C`** they physically **cannot dominate** geometry once \(C\) is small.

Objective glue (primal hinge view): schematically,

$$
\text{Total Cost}
\approx
\tfrac12\|\mathbf{w}\|^{2} \;+\; C\sum_{i} \bigl[\text{(hinge or slack surrogate on row } i)\bigr].
$$

**High \(C\)** screams “almost no hinge allowed,” encouraging **narrow, rigid** separators that chase training noise. **Low \(C\)** cheapens hinge violations—it’s cheaper to tolerate a rogue cluster than inflate \(\mathbf{w}\) without bound.

\(C\) is also the \(\alpha\) **upper bound** in the dual, so outliers **bounce off the \(C\) ceiling** instead of cranking \(\alpha\to\infty\) and pinning the corridor.

---

### 3. Geometric effect on the “street”?

**Usually a visibly wider soft margin:** lowering \(C\) makes it comparatively cheap to leave training points inside the corridor or mildly misclassified, so the solver stops pinching boundaries around bad sensors solely to chase zero hinge. You trade **narrow perfect separation** on the training scrape for **stability.** **High \(C\)** favors **narrow** margins and brittle fits; **low \(C\)** favors **thickening** viable corridor geometry under the \(\alpha\le C\) cap on outlier screaming.

---

### Quick sanity recap

| Change | Typical effect |
|--------|----------------|
| **↓ C** (lower **C**) | Wider tolerated margin intrusion, outliers capped (**α ≤ C**) |
| **↑ C** (raise **C**) | Margin shrinks chasing training fidelity (overfitting risk) |

Use validation curves / ROC / calibrated holdout—not gut feel alone—to pick \(C\) on real fraud telemetry.
