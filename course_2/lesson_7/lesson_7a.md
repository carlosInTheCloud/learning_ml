# Course 2 — Lesson 7 — Decision Tree Factories

As we move from the **Equation Family** (logistic regression, SVMs) into the **Rule Family**, it helps to reset the mental model. We stop drawing smooth boundaries with dot products, gradients, and kernels. Instead, we play a game of **discrete, sequential questions** — like *20 Questions*, but with provable criteria for which question to ask next.

This lesson builds in layers:

1. **Philosophy of the Rule Family** — axis-parallel splits and flowcharts vs. global equations.
2. **Entropy & Information Gain** — how the algorithm scores "good" splits.
3. **Leaf predictions** — mode for classification, mean for regression.
4. **Production trade-offs** — no scaling, explainability, and the overfitting trap.
5. **Bagging & Random Forests** — bootstrap variance reduction, OOB samples, feature subsampling, parallelism, plus a **worked three-tree vote** on a tiny cycling dataset.
6. **Boosting (GBM / XGBoost)** — sequential bias reduction via pseudo-residuals and gradient descent; a **hand-worked boosting step**; **learning rate vs. `n_estimators`** and how to tune in practice.

---

## Part 1: Philosophy of the Rule Family

In the Equation Family, models use **continuous** math ($\mathbf{w} \cdot \mathbf{x} + b$) and weigh **all features at once**. With ten features, an SVM assigns a weight to every dimension and produces a single score.

The Rule Family does **not** use one continuous equation for the whole input. It applies **discrete, sequential logic**: look at features **one at a time** and build a **flowchart of binary rules** (if/then).

### Axis-parallel splits

Geometrically, a decision tree does **not** draw a smooth diagonal separator. It **chops** the feature space with **axis-aligned** cuts — horizontal and vertical in 2D — producing **rectangular** regions (boxes).

**Example (cycling features):**

1. First split: **Power > 250 W?** → vertical boundary in the power dimension.
2. Inside the high-power region: **Cadence < 80?** → horizontal boundary in the cadence dimension.

The algorithm keeps subdividing until each box is **pure** enough (e.g., only bonk rides or only no-bonk rides), subject to stopping rules.

---

## Part 2: How the tree chooses a split — entropy & information gain

Which feature should we split on first, and where should the threshold sit? Intuition: a **good first question** eliminates a lot of uncertainty (like "Is it a mammal?"), not a question that barely narrows things down (like "Does it have exactly 13 spots?").

The standard formalism uses **Shannon entropy**.

### 1. Shannon entropy $H$

Entropy measures **uncertainty / impurity** in a subset of labels.

$$
H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)
$$

- $S$ — current subset of data.
- $c$ — number of classes (e.g., 2 for Bonk / No Bonk).
- $p_i$ — proportion of class $i$ in $S$.

**Reading the equation:**

- **Maximum chaos (binary, 50/50):** $p_{\text{bonk}} = 0.5$, $p_{\text{no}} = 0.5$

  $$
  H = -\bigl(0.5 \log_2 0.5 + 0.5 \log_2 0.5\bigr) = 1.0
  $$

- **Pure node:** 100% one class → one $p_i = 1$, others 0 (with the convention $0 \log 0 = 0$) → **$H = 0$**.

### 2. Information gain $\mathrm{IG}$

For a candidate split on attribute $A$, compare entropy **before** the split to the **weighted average** entropy **after**.

$$
\mathrm{IG}(S, A) = H(S) - \sum_{v \in \mathrm{Values}(A)} \frac{|S_v|}{|S|}\, H(S_v)
$$

**Translation:** Information gain = (parent chaos) − (weighted average chaos of children).

In basic greedy tree learning, the algorithm tries candidate splits (often over feature thresholds), computes information gain for each, and picks the split with the **highest** gain — the split that **removes the most uncertainty**.

> **Sidebar — Information theory**  
> Claude Shannon's information theory underpins much of the digital world (compression, channels, coding). A useful follow-up for your notes: *why entropy uses $\log_2$, and how it relates to "bits" of information.*

---

## Part 3: Predictions at the leaves

Unlike an SVM score $\mathbf{w} \cdot \mathbf{x} + b$, a fitted tree **routes** a new example down the tree to a **leaf**.

- **Classification:** predict the **majority class** in that leaf (the **mode**). If the leaf has 8 bonks and 2 no-bonks, predict "bonk" — often summarized as a **Vote fraction** / probability estimate from class counts.
- **Regression:** predict the **mean** of the target in the leaf (e.g., average heart rate of training points that land there).

---

## Part 4: Production strengths — and the fatal flaw

### Strengths

1. **No feature scaling required** — splits depend on **orderings** along axes, not on dot-product geometry. Power in hundreds and cadence in tens do not need normalization for the split to "work."
2. **Strong explainability** — you can print the **exact path** of if/then rules; useful for audits and regulated domains.

### Fatal flaw: unconstrained depth → high variance

If allowed to grow without limit, a tree can keep splitting until **every leaf holds a single training point** (entropy zero on the training set). It **memorizes** noise — including a single weird ride from a sensor glitch. A **single deep tree** is the textbook picture of **high variance / overfitting** and is usually **not** production-ready alone.

**Mitigations:** **pruning** / depth limits / min samples per leaf — or **ensembles** that average or sequence many trees (Random Forests, boosting).

---

## Part 5: Bagging — Random Forests

A lone deep tree often **overfits**. **Bagging** (*bootstrap aggregating*) reduces **variance** by training many trees on **randomized** views of the data and **aggregating** their predictions.

### Mental model: diverse "board members"

If every committee member read the **same** report, they would make the **same** mistakes — predictions are **correlated**. For "wisdom of the crowd," inject **randomness**:

- Give each tree a **different bootstrap sample** of the training rows.
- Restrict which features each split may consider (**random feature subsets**).

At inference, trees **vote** (classification) or **average** (regression); individual quirks tend to **cancel**.

### Bootstrapping & out-of-bag (OOB) error

Draw $N$ training rows **with replacement** to form one tree's training set. Some rows appear **multiple** times; some **never** appear in that bootstrap sample.

The probability that a **specific** row is **omitted** from one bootstrap sample, in the large-$N$ limit, is:

$$
\lim_{N \to \infty} \left(1 - \frac{1}{N}\right)^N = \frac{1}{e} \approx 0.368
$$

So roughly **63.2%** of **unique** rows appear in a given tree's bag; about **36.8%** are **out-of-bag** for that tree. OOB rows can be used as a **built-in validation** mechanism across trees.

### Feature subsampling at each split

When expanding a node, the algorithm often does **not** search all $M$ features. A common heuristic is to sample about

$$
m \approx \sqrt{M}
$$

features to evaluate per split (e.g., 4 random features out of 16). That **decorrelates** trees.

### Architect's note: parallelism

Tree 1 does not depend on Tree 99 during training. Random Forest training is **embarrassingly parallel** — one core per tree is a natural pattern (library details aside).

### Worked example — Random Forest with three trees

**Goal:** Predict **Bonk (Y/N)** from **Power (W)**, **Sleep (hrs)**, **Temp (°F)**.

#### Master dataset ($N = 5$)

| Ride ID | Power (W) | Sleep (hrs) | Temp (°F) | Bonk? |
|--------|-----------|-------------|-----------|-------|
| R1     | 200       | 8           | 70        | No    |
| R2     | 310       | 5           | 95        | Yes   |
| R3     | 240       | 7           | 75        | No    |
| R4     | 305       | 8           | 85        | Yes   |
| R5     | 280       | 4           | 72        | Yes   |

#### Step 1 — Bootstrap bags (with replacement)

| Tree | Bootstrap draw (5 rows) | Notes |
|------|-------------------------|--------|
| 1    | R1, R2, R2, R4, R5      | R2 twice; **R3 OOB** |
| 2    | R1, R1, R3, R4, R5      | **R2 OOB** |
| 3    | R2, R3, R4, R4, R4      | R4 repeated; **R1, R5 OOB** |

#### Step 2 — Feature subsampling ($m = 2$ of 3 features)

| Tree | Features allowed |
|------|------------------|
| 1    | Power, Sleep     |
| 2    | Sleep, Temp      |
| 3    | Power, Temp      |

#### Step 3 — Stylized rules (illustrative weak learners)

**Tree 1** (bag 1, Power & Sleep): IF **Power > 270** → **Yes (bonk)**; else → No.

**Tree 2** (bag 2, Sleep & Temp): IF **Sleep < 5** → **Yes**; else → No.

**Tree 3** (bag 3, Power & Temp): IF **Temp > 80** → **Yes**; else → No.

Individually these rules are **biased** (e.g., Tree 2 ignores power); ensemble diversity is the point.

#### Step 4 — Inference vote

**New ride:** Power = **290** W, Sleep = **7** h, Temp = **82** °F.

- Tree 1: 290 > 270 → **Yes**
- Tree 2: 7 $\nless$ 5 → **No**
- Tree 3: 82 > 80 → **Yes**

**Aggregate:** 2 Yes, 1 No → predict **Bonk** (~**66.7%** vote share for Yes).

**Takeaway:** Bootstrapping hid some extreme rows from specific trees; feature subsampling **decorrelated** their mistakes so **biased** individual trees were **outvoted** where appropriate.

---

## Part 6: Boosting — GBM / XGBoost / LightGBM

**Boosting** targets **bias** and **sequential** refinement: many **shallow**, **weak** learners chained so each new tree fixes **what the ensemble still gets wrong**.

### Mental model: chain of specialists

Picture **sequential** golf shots:

1. First shallow tree predicts bonk vs. not; it **misses** by a margin.
2. The **next** tree is trained to predict the **remaining error** (not the raw labels from scratch).
3. Each addition nudges the prediction; with a small **learning rate**, updates are **conservative** to avoid overshooting.

### Pseudo-residuals & gradient descent

Formally, boosting minimizes a **loss** $L(y, F(\mathbf{x}))$ over an additive model $F$. At stage $m$, the **pseudo-residual** for example $i$ is (schematically) the **negative gradient** of the loss with respect to the current prediction:

$$
r_{im} = -\left[\frac{\partial L\bigl(y_i, F(\mathbf{x}_i)\bigr)}{\partial F(\mathbf{x}_i)}\right]_{F(\mathbf{x}) = F_{m-1}(\mathbf{x})}
$$

A new tree $h_m$ is fit to these residuals (or to binned approximations in libraries). The model updates as

$$
F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu\, h_m(\mathbf{x})
$$

where $\nu$ is the **learning rate** (often called **eta** in XGBoost).

### Architect's note: sequential bottleneck

Boosting steps are inherently **ordered**: tree 50 needs **updated targets / residuals** after tree 49. You cannot trivially parallelize **across boosting stages** the way you parallelize Random Forest trees. (Libraries still parallelize **within** a tree or other sub-tasks.)

### Ensemble comparison (one-line memory)

| Idea | Typical trees | Training | Main fight |
|------|----------------|----------|------------|
| **Bagging (RF)** Deep / full | Parallel | **Variance** (overfitting noise) |
| **Boosting** | Shallow, many | Sequential | **Bias** + residual shape via gradients |

### Worked example — one gradient boosting step (classification)

Targets: **Bonk = 1**, **No bonk = 0**.

| Ride | Power | Sleep | Temp | $y$ |
|------|-------|-------|------|-----|
| R1   | 200   | 8     | 70   | 0   |
| R2   | 310   | 5     | 95   | 1   |
| R3   | 240   | 7     | 75   | 0   |
| R4   | 305   | 8     | 85   | 1   |
| R5   | 280   | 4     | 72   | 1   |

#### Step 0 — Baseline $F_0$ (log odds)

Class proportion: 3 bonks / 5 rows → $p = 0.6$.

$$
F_0(\mathbf{x}) = \ln\left(\frac{p}{1-p}\right) = \ln\left(\frac{0.6}{0.4}\right) \approx 0.405
$$

Every row starts with **baseline** bonk probability **0.6**.

#### Step 1 — Pseudo-residuals (probability residual)

For each row: $r = y - p$ with $p = 0.6$.

| Row | $y$ | Residual |
|-----|-----|----------|
| R1  | 0   | $-0.6$   |
| R2  | 1   | $+0.4$   |
| R3  | 0   | $-0.6$   |
| R4  | 1   | $+0.4$   |
| R5  | 1   | $+0.4$   |

#### Step 2 — Tree 1 fits residuals

Tree 1 is trained to predict **these residuals**, not the raw 0/1 labels. Illustrative best split: **Power > 250**.

- Left leaf (Power $\le$ 250): R1, R3 — residuals $-0.6, -0.6$
- Right leaf (Power > 250): R2, R4, R5 — residuals $+0.4, +0.4, +0.4$

#### Step 3 — Leaf values $\gamma$ (log-odds correction)

For logistic-style boosting, leaf updates often use a **Newton**-style ratio (here with a common $p(1-p) = 0.6 \times 0.4 = 0.24$ for all points at this stage):

$$
\gamma = \frac{\sum \text{residuals}}{\sum \bigl[p(1-p)\bigr]}
$$

- Left: $\displaystyle \gamma_{\text{left}} = \frac{-1.2}{0.48} = -2.5$
- Right: $\displaystyle \gamma_{\text{right}} = \frac{1.2}{0.72} \approx 1.67$

#### Step 4 — Update with learning rate $\nu = 0.1$

$$
F_1(\mathbf{x}) = F_0(\mathbf{x}) + \nu \cdot \gamma_{\text{leaf}(\mathbf{x})}
$$

- R1, R3: $0.405 + 0.1 \times (-2.5) = 0.155$ → $p = \dfrac{e^{0.155}}{1+e^{0.155}} \approx 0.538$
- R2, R4, R5: $0.405 + 0.1 \times 1.67 \approx 0.572$ → $p \approx 0.639$

**After one small step:** probabilities for **true 0s** **drop** from 0.6 toward 0; for **true 1s** they **rise** — errors shrink. Repeating the loop with new residuals trains Tree 2, etc. Production models run **hundreds or thousands** of such stages with tuned $\nu$.

### Learning rate $\nu$ vs. `n_estimators`

#### Parameters vs. hyperparameters

- **Parameters** (learned from data): split locations, leaf values $\gamma$, structure of each tree.
- **Hyperparameters** (set before training): **learning rate** $\nu$, **number of trees** (`n_estimators`), max depth, min samples per leaf, subsampling fractions, etc. The data **do not** tell you $\nu$ — you (or search) choose it.

#### The see-saw: large $\nu$ vs. small $\nu$

| $\nu$ | Effect | Trees needed | Risk |
|-------|--------|--------------|------|
| **High** (e.g. 0.5) | Large corrective steps | Fewer trees | **Overshoot** oscillation; worse minima |
| **Low** (e.g. 0.01) | Tiny steps | **Many** trees | **Slow**, costly training; usually stabler path |

#### Production tuning sketch

1. Fix a **compute budget** (e.g. cap `n_estimators` at what training time allows).
2. **Search** $\nu$ (often on a **log grid**) with cross-validation for that budget.
3. Treat **too-fast convergence** with high $\nu$ as suspicious — check validation curves for instability.

**Intuition:** $\nu$ is a **brake pedal** on how aggressively each tree corrects residuals; smaller $\nu$ often yields **better generalization** if you **buy** enough trees to compensate.

---

## Closing

You now have the core map: **trees** split to **reduce entropy** (information gain); **forests** **bag** deep trees to fight **variance**; **boosters** chain shallow trees on **gradients** of the loss to fight **bias**, with **$\nu$** trading step size against **tree count**. When that lines up with how you think about cycling features and production constraints, the next step is **implementing and tuning** these models in code.
