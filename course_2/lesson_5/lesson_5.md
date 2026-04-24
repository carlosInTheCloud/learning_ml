# Course 2 — Lesson 5, Part 1 — Entropy & information gain

## Introduction to the Rule Family: Decision Trees & Ensembles

Now that you have mastered the Equation Family (Linear Regression, Logistic Regression, and Neural Networks), we are going to shift gears and explore the Rule Family: Decision Trees, Random Forests, and XGBoost.

While these two families "think" differently—equations draw smooth, continuous curves using calculus, while trees draw rigid, step-by-step boxes using discrete logic—they both belong to the exact same toolbox: Supervised Machine Learning.

Here is a high-level overview of how these tree-based algorithms actually work.

### The Two-Phase Engine

In supervised learning, it can feel counterintuitive to feed an algorithm the "answers" to a problem. But we don't build these trees to predict the past; we use the past to write a rulebook for the future. This happens in two distinct phases:

#### Phase 1: Training (Building the Rulebook)

During training, we give the algorithm historical data where we already know the outcome.

- **The Features ($X$):** The historical clues (e.g., Power and Cadence data from past rides).
- **The Ground Truth ($y$):** The actual, known outcome (e.g., Bonk or No Bonk).

The algorithm's job is not to guess $y$, but to use it as an answer key. It plays a massive, mathematical game of "20 Questions." It tests every possible threshold to find the exact rules—like "Is Power $\ge 230\text{W}$?"—that best separate the data into pure buckets. Once it finishes building this flowchart, Phase 1 is done, and the historical $y$ answers are discarded.

#### Phase 2: Inference (Predicting the Unknown)

This is where the tree is deployed to generate actual value.

- **New Data (New $X$):** We feed the tree brand new data for an upcoming event (e.g., planning a new ride at $240\text{W}$).
- **The Prediction ($\hat{y}$):** Because this event hasn't happened yet, there is no $y$. Instead, the new data drops into the top of the flowchart built in Phase 1. It follows the logic gates down the branches until it hits a final bucket (a **Leaf Node**), which outputs the official prediction.

By operating this way, Decision Trees give us a highly interpretable, "white box" model. We can trace exactly why the algorithm made a specific prediction, setting the perfect foundation before we scale them up into massive Ensembles like Random Forests.

## Zooming In: The Anatomy of a Decision Tree

Before we can combine hundreds of models into a massive ensemble like a Random Forest, we need to understand how to grow just one. At its core, a single Decision Tree is simply a machine-generated flowchart.

Unlike a human expert who might weigh dozens of pros and cons simultaneously to make a decision, a Decision Tree is strictly binary and sequential. It looks at the dataset and asks a single yes/no question to split the data into two smaller buckets.

It then moves into those new buckets and asks another question, repeating this process recursively until it reaches a definitive conclusion (a final **Leaf Node**).

But how does the computer know which question to ask first? It doesn't use intuition; it uses pure mathematical brute force. It calculates the **entropy** (the amount of chaos or uncertainty) in the current dataset, and tests every possible split to find the one that yields the highest **information gain** (the most chaos destroyed).

By ruthlessly picking the best mathematical split at every single step—a process known as a **greedy algorithm**—the tree automatically writes the optimal, most efficient rulebook for your data.

---

To build a **decision tree**, the computer must decide **which question to ask first** at the top of the flowchart.

Imagine you want to predict whether a ride ends in a **“Bonk.”** You have two candidate splits:

- Was **power** > 200 W?
- Did you **sleep** > 8 hours?

**How does the math pick the better starting question?** It looks for the question that removes the most **chaos** (uncertainty) in the labels.

To quantify chaos, machine learning borrows **entropy** from information theory (and related ideas in physics and telecommunications).

---

## 1. Shannon entropy ($H$)

Claude Shannon introduced this in 1948. **Entropy** measures **impurity**, **uncertainty**, or **surprise** in a set of labels.

- If every example has the **same** label (e.g. 100% Bonk), there is **no surprise** → entropy is **$0$** (pure).
- If labels are **evenly split** (e.g. 50% Bonk / 50% No Bonk), uncertainty is **maximal** for a binary problem → entropy is **$1$** bit (using the usual binary entropy scale below).

---

## 2. The formula (binary labels)

For a finite set $S$ with **binary** outcomes (positive / negative), **Shannon entropy** (base 2) is:

$$
H(S) = -p_{+} \log_2(p_{+}) - p_{-} \log_2(p_{-})
$$

where:

- **$p_{+}$** = proportion of **positive** examples (e.g. Bonk),
- **$p_{-}$** = proportion of **negative** examples (e.g. No Bonk),
- **$\log_2$** = logarithm base 2, so entropy is measured in **bits**.

**Convention:** If $p_{+} = 0$ or $p_{-} = 0$, treat **$0 \log_2 0$** as **$0$** (limit), so pure nodes have entropy $0$.

**Note:** In **deep learning**, we often use the **natural logarithm** ($\ln$) because it pairs cleanly with calculus and gradients. For **discrete** splits and decision trees, **$\log_2$** is standard so entropy is in **bits** and peaks at $1$ for a 50/50 binary split.

---

## 3. A concrete example (Lab 4–style data)

Use the same tiny setup as in **Lesson 4**’s lab: **$m = 4$** rides.

- **1** ride: Bonk (positive)
- **3** rides: No Bonk (negative)

### Step A — Probabilities

$$
p_{+} = \frac{1}{4} = 0.25, \qquad p_{-} = \frac{3}{4} = 0.75
$$

### Step B — Plug into the formula

$$
H = -(0.25 \times \log_2(0.25)) - (0.75 \times \log_2(0.75))
$$

Useful values:

- $\log_2(0.25) = -2$
- $\log_2(0.75) \approx -0.415$

So:

$$
H = -(0.25 \times -2) - (0.75 \times -0.415) = 0.5 + 0.311 \approx \mathbf{0.811}
$$

### Interpretation

The entropy of this starting set is about **$0.811$** bits. That is **fairly high** (near the maximum of **$1$** for a binary split), so the set is **impure** and **uncertain**: the labels are mixed.

When we build a **decision tree**, we will choose splits (e.g. “Is power high?”) that divide the data into **smaller groups** whose entropy moves **toward $0$** (purer leaves).

The **drop** in entropy from the parent set to the children after a split is quantified by **information gain** (below).

---

## Part 1B — Information gain (IG)

If **entropy** measures **chaos**, **information gain** measures **how much chaos we remove** by asking a **specific** question (a candidate split).

A decision tree algorithm is like a long game of **“20 questions.”** It tries many possible splits, for example:

- “What if I split on **power > 200 W**?”
- “What if I split on **cadence > 85 rpm**?”

For **each** candidate split, it computes **information gain**. The split with the **highest** gain (largest reduction in weighted chaos) is a strong choice for the next **node** in the tree.

### The information gain formula

Compare entropy **before** the split (the **parent** node) to the **weighted average** entropy **after** the split (the **child** buckets):

$$
\text{IG} = H(\text{Parent}) - \left( \frac{m_{\text{left}}}{m}\, H(\text{Left}) + \frac{m_{\text{right}}}{m}\, H(\text{Right}) \right)
$$

where:

- **$H(\text{Parent})$** — entropy of **all** examples at the node **before** splitting.
- **$m$** — total number of examples in the parent node.
- **$m_{\text{left}}, m_{\text{right}}$** — how many examples go to the **left** vs **right** child.
- **$H(\text{Left}), H(\text{Right})$** — entropy of the label distribution **inside** each child.

The **weights** $m_{\text{left}}/m$ and $m_{\text{right}}/m$ matter: we care about **overall** impurity across the **whole** dataset mass, not only a tiny pure leaf. A split that creates one **huge, still-chaotic** bucket and one **tiny** pure bucket can score poorly if the weighted average entropy stays high—so the formula **penalizes** lopsided splits that do not genuinely clean up most of the data.

---

## Part 1C — Worked example: “Was Power High?”

Our **parent** dataset is perfectly split: **3 Bonks** and **3 No Bonks**. Because it is a 50/50 coin toss, the starting chaos is at its **absolute maximum**:

$$
H(\text{Parent}) = \mathbf{1.0}
$$

We want to test whether asking **“Was Power High?”** is a good split. Here is the data table (“Bonk?” = **Yes** means Bonk, **No** means No Bonk):

| Ride | Power | Bonk? |
| ---: | :---: | :---: |
| 1 | High | Yes |
| 2 | High | Yes |
| 3 | High | Yes |
| 4 | High | No |
| 5 | Low | No |
| 6 | Low | No |

### Step 1 — Execute the split

Imagine physically sorting these **6** rides into two buckets based on **“Is Power High?”**

**Left bucket (Yes — High power):** rides **1, 2, 3, 4** → **4** rides total: **3 Bonks**, **1 No Bonk**.

**Right bucket (No — Low power):** rides **5, 6** → **2** rides total: **0 Bonks**, **2 No Bonks**.

### Step 2 — Entropy of the children

**1. Left bucket — $H(\text{Left})$**

Total rides: **$4$**.

$$
p_{+}(\text{Bonk}) = \frac{3}{4} = 0.75, \qquad p_{-}(\text{No Bonk}) = \frac{1}{4} = 0.25
$$

This is the same label mix as in **section 3** above, so:

$$
H(\text{Left}) = -(0.75 \log_2(0.75)) - (0.25 \log_2(0.25)) \approx \mathbf{0.811}
$$

**2. Right bucket — $H(\text{Right})$**

Total rides: **$2$**.

$$
p_{+}(\text{Bonk}) = \frac{0}{2} = 0.0, \qquad p_{-}(\text{No Bonk}) = \frac{2}{2} = 1.0
$$

Because this bucket is **100% pure** (only No Bonks), we know the answer without extra calculation:

$$
H(\text{Right}) = \mathbf{0.0}
$$

### Step 3 — Weighted average entropy

We cannot simply add **$0.811$** and **$0.0$**. The left bucket has twice as many rides as the right, so its chaos counts more. Weight by size relative to the parent (**$m = 6$**):

$$
\text{Weighted Entropy} = \left( \frac{4}{6} \times 0.811 \right) + \left( \frac{2}{6} \times 0.0 \right)
$$

$$
\text{Weighted Entropy} = (0.667 \times 0.811) + 0 \approx \mathbf{0.541}
$$

This is the **total chaos remaining** after splitting on Power.

### Step 4 — Information gain (IG)

Subtract remaining chaos from starting chaos:

$$
\text{IG} = H(\text{Parent}) - \text{Weighted Entropy}
$$

$$
\text{IG} = 1.0 - 0.541 = \mathbf{0.459}
$$

### Conclusion

By asking **“Was Power High?”**, we remove about **$0.459$ bits** of chaos.

In code, the algorithm repeats this calculation for **every other feature** (e.g. “Was cadence high?”, “Was duration > 2 hours?”). Whichever split yields the **highest information gain** is chosen as the **first split** at the **root** of the decision tree.

---

## Part 2 — Two families: ID3 / C4.5 vs CART

Historically, there are two major **families** of decision-tree algorithms (often introduced in a syllabus as **ID3** and **CART**). Here is how they treat a feature like power when it has **High**, **Medium**, and **Low** categories.

### 1. Multi-way splits (ID3 / C4.5)

Algorithms in the **ID3** family (including extensions such as **C4.5**) allow a node to split into **as many child buckets as there are categories**.

If power takes values High, Medium, and Low, the tree grows **three branches at once** from that node.

**Information gain** uses the same idea as before: compare parent entropy to a **weighted average** of child entropies. With three children, you subtract **three** weighted terms instead of two—one weight $m_i / m$ per bucket, each multiplied by $H(\text{child}_i)$.

**Drawback:** Multi-way splits tend to **shatter** the data quickly. A categorical field with many levels (e.g. **20 cities**) creates **20 small buckets** in one step, which often drives **severe overfitting**.

### 2. Strictly binary splits (CART)

**CART** stands for **Classification and Regression Trees**. It is the usual choice in modern practice and is what libraries such as **scikit-learn** use for their standard tree learners.

CART **always** uses **binary** splits: exactly **two** children per internal node. For ordered or nominal categories, it does this by asking **yes/no** questions about **subsets** of the levels.

For High, Medium, and Low power, CART might first ask:

**“Is power High?”**

- **Left (Yes):** only High-power examples.
- **Right (No):** Medium **and** Low together.

Deeper in the tree, on that right branch, a later split might ask **“Is power Medium?”**, separating Medium from Low.

**Advantage:** Binary splits produce **deeper, more gradual** trees, reduce one-step shattering, and allow the **same feature** to be used again at lower depths with a different threshold or subset.

---

**Takeaway:** The math of information gain generalizes to **any number** of child buckets, but many production systems **deliberately restrict** each split to **two** branches so trees stay **more stable** and **generalize** better.

---

## Part 3 — Overfitting and pruning

If you run a **standard CART** learner on your cycling data with **no extra constraints**, it will keep splitting until **every leaf** has label entropy **exactly $0.0$**—pure buckets only.

That can produce absurdly specific rules, along the lines of: “Was cadence > 82 and power > 190 and temperature = 72 and did you have exactly 1.5 scoops of drink mix?” The tree can reach **100% accuracy on the training set**, yet **fall apart on new rides**: it has **memorized** past examples, not **stable** patterns of cycling.

This is classic **high variance** (**overfitting**). The main remedy is **pruning**—limiting complexity either **while** the tree grows or **after** it has grown.

### 1. Pre-pruning (early stopping)

This is the approach you see most often in practice: **hyperparameters** cap growth **before** the tree becomes a dictionary of the training rows.

- **Max depth:** e.g. “At most **3** successive questions from the root.” When depth hits the limit, splitting stops even if some nodes are still **impure** (entropy **> 0**).
- **Minimum samples per split:** e.g. “Do not split a node that holds **fewer than 10** rides.” That blocks hyper-specific rules for **tiny** groups and outliers.
- **Minimum information gain:** e.g. “If the best candidate split has **IG < 0.05**, do **not** split.” That suppresses **weak** questions that barely reduce chaos.

### 2. Post-pruning (grow, then cut)

Here the tree is allowed to grow **large** (often fully fitting the training noise). Then, using a **validation** set (or a cost–complexity criterion), you walk **from the leaves upward** and ask whether each **branch** actually **helps** out-of-sample performance. If removing a subtree **does not hurt** (or **helps**) validation quality, you **remove** it. You repeat until further cuts would **clearly** worsen the model.

Think of it as trimming the tree **after** seeing how much each twig contributes on **fresh** data.

### Connection to Lessons 3 and 4

**Forcing a simpler model** should sound familiar. In **Lessons 3 and 4**, **$L2$ regularization** (the **$\lambda$** penalty on large weights) **constrains** a neural network so it cannot chase every quirk of the training set.

**Pruning** is the same **bias–variance** idea for **trees**: both are tools to **fight high variance** by trading a bit of flexibility on the training data for **better generalization**.

---

## Part 4 — Ensembles: bagging and boosting

This is where much of **modern applied** machine learning lives. In a **Kaggle**-style competition, or when shipping a **tabular** predictive model inside a product, you will very often use an **ensemble**.

An ensemble uses **wisdom of the crowd**: instead of one **deep, unstable** decision tree, you train **many** trees (hundreds or thousands) and **combine** their predictions.

There are two dominant coordination patterns: **bagging** and **boosting**.

### 1. Bagging (bootstrap aggregating)

**Bagging** stresses **parallel** models whose errors are **averaged out**—primarily attacking **variance** (**overfitting**). The flagship example is the **random forest**.

**How it works**

You might train **1,000** trees **independently**. If every tree saw the **identical** training set, they would often learn **nearly the same** splits and repeat the **same** mistakes. Bagging breaks that symmetry with **randomness**:

- **Row randomness (bootstrapping):** each tree fits a **resampled** subset of training rows (with replacement), so no two trees see exactly the same data.
- **Feature randomness:** at each split, the algorithm considers only a **random subset** of features—not the full list. One tree might mainly see **cadence** and **duration** at a node; another might see **power** and **heart rate**.

**Intuition**

Picture debugging a **phantom shifting** problem on a drivetrain. One expert staring at **everything** might lock onto a **narrow, wrong** story. If **many** mechanics each focus on **one or two** subsystems—cables, cassette, hanger—and you **aggregate** their conclusions, the **consensus** is often **stabler** and **more accurate** than any single “full picture” diagnosis.

**Result**

Many **moderately simple** trees cast a **majority vote** (for classification) or an **average** (for regression). **Idiosyncratic** errors tend to **cancel**; you get strong **variance reduction** without hand-tuning a single giant tree’s prune settings.

### 2. Boosting

**Boosting** stresses **sequential** correction—each new model targets what the **previous** ones got wrong. That pushes **accuracy** hard and mainly fights **underfitting** and **residual error** (**bias** in a loose sense). Well-known methods include **AdaBoost** and **gradient boosting** (a family that includes the widely used **XGBoost** implementation).

**How it works**

Trees are built **one after another**, not in parallel.

- **Tree 1** fits the data and makes predictions; some examples are **wrong**.
- **Tree 2** is trained on a **reweighted** dataset: misclassified (or large-error) points count **more**, so the new tree is pushed to **fix** what Tree 1 missed.
- **Tree 3** targets what still remains after Tree 2—and so on.

**Intuition**

It resembles **periodized training**: a baseline test shows **VO₂ max** is the limiter, so the next block **targets** that weakness. After a retest, **threshold power** might become the weak link, so the **following** block addresses **that**. Each phase attacks the **residual** gap left by the last.

**Result**

You get a **chain** of small corrections. **Gradient boosting** and its optimized implementations are often the **strongest default** for **structured / tabular** data—while **watching** validation performance, because a **very long** chain can still **overfit**.

### Summary comparison

| | **Bagging (e.g. random forest)** | **Boosting (e.g. gradient boosting / XGBoost)** |
| --- | --- | --- |
| **Execution** | Parallel: many trees trained **at once** | Sequential: trees built **one after another** |
| **Primary goal** | Cut **variance** (stabilize, reduce overfitting) | Cut **bias** / residual error (drive up **accuracy**) |
| **Robustness** | Often **very stable**; hard to overfit with defaults alone | **Can** overfit if the chain is **too** deep or steps are unchecked |

These are the **conceptual skeleton** of the tree-based methods that underpin a large share of **tabular** ML in industry and competitions—single trees, **pruning**, then **ensembles** that **bag** or **boost** them.

---

## Part 5 — Brute force at each node: who asks the questions?

The secret is that the algorithm has **no intuition**. It does not “think up” good questions. It uses **exhaustive search** over mechanically generated candidates—**brute force** guided by the **information gain** score.

Here is **who** proposes the splits and **how** they are built.

### 1. Who formulates the questions?

**No human** hand-writes the rules. A learner such as **CART** **generates** candidate questions from the **values actually present** in each column: thresholds for numbers, equality tests for categories, and so on.

### 2. How does it formulate them?

The recipe depends on whether the feature is **categorical** (labels) or **continuous** (real numbers).

**Categorical data** (e.g. cadence: High, Low)

The procedure inspects **distinct** categories and builds **binary** tests such as:

- **Question A:** Is cadence **==** High?
- **Question B:** Is cadence **==** Low?

**Continuous data** (e.g. power: 190 W, 200 W, 215 W)

A spectrum is handled with **inequalities**, not “equals 200 W” as the only option:

1. **Sort** the observed values: e.g. **[190, 200, 215]**.
2. Consider **midpoints** between **adjacent** sorted values: e.g. **195** and **207.5**.
3. For each midpoint $t$, add a candidate such as: **Is power > $t$?**

So you get:

- **Question C:** Is power **>** 195?
- **Question D:** Is power **>** 207.5?

If a column has **10,000** **distinct** numeric values, there are **9,999** adjacent gaps—hence **9,999** threshold tests of that form (implementations often **prune** redundant or duplicate candidates for speed, but the **idea** is this exhaustive grid).

### 3. The master loop (building one node)

After generating **all** candidate questions for **every** column (within the rules of the implementation), the algorithm runs the same **evaluation loop** at **each** node:

1. For **each** candidate split: partition the rows, compute **child entropies**, then **information gain**.
2. **Keep** the candidate with the **largest** gain—that split becomes the **fixed** rule for **this** node.
3. **Recurse** on the **left** and **right** child subsets: repeat the whole process **only** on the data that fell into each bucket.

You can picture an inner monologue at the root: *“Test cadence == High—record IG. Test power > 195—higher IG? replace best-so-far. Test power > 207.5…”* After every candidate is scored, the **winner** is installed; then the same **brute-force** search runs **inside** each new child.

### Takeaway

Tree induction at each depth is typically **greedy**: it does **not** plan several moves ahead. It only maximizes **immediate** impurity reduction (or gain), then moves down and repeats. That locality is simple and fast—but it is why **depth limits**, **pruning**, and **ensembles** matter for **generalization**.
