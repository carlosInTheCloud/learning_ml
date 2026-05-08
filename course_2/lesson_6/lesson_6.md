# Course 2 — Lesson 6 — Support Vector Machines (Parts 1 & 2)

To tie this back to our **algorithm toolbox**, Support Vector Machines (SVMs) belong to the **Equation Family**. Like logistic regression, they use continuous math to draw a line (or a flat plane) through your data. The way they choose *where* to draw that line, however, is distinct.

Below is **Part 1**: the maximum-margin idea and where the name “support vector” comes from. **Part 2** connects the Lagrangian dual to the **kernel trick**—where SVMs stop being “only geometry” and become a famously efficient way to capture **nonlinear** boundaries.

---

## Part 1: The maximum margin classifier

Imagine plotting cycling data on a standard 2D graph: **power** on the *x*-axis and **cadence** on the *y*-axis. Rides where you **bonked** are red dots; rides where you **felt great** are blue dots.

Your goal is to draw a **straight line** that **perfectly separates** the red dots from the blue dots.

When there is a clear gap between the two clusters, there is not just one line that works—there are **infinitely many**. You could hug the red cluster, hug the blue cluster, or pick some diagonal that still keeps the colors on opposite sides.

**Logistic regression** picks a boundary by minimizing its loss; that boundary can end up **uncomfortably close** to one cluster.

An **SVM** asks a different question:

> Which line gives us the **widest possible margin of safety**?

---

### The “street” analogy

Instead of only a thin separator, think of an SVM as trying to paint a **wide, multi-lane “street”** between the two clusters.

- The **center line** of the street is the **decision boundary** (the actual classifier).
- The **outer edges** of the street (the shoulders) grow until they **touch** the closest red point on one side and the closest blue point on the other.

The algorithm **rotates and shifts** this street until the **width** of the corridor is as large as geometry allows. That is **maximizing the margin**.

---

### What are “support vectors”?

Hence the algorithm’s name.

In a dataset of **10,000** historical rides, the striking fact is that an SVM effectively **ignores almost all** of the points.

- It does not need the deep-interior red points (huge bonks far inside the red region).
- It does not need the deep-interior blue points (easy rides far inside the blue region).

It **only** needs the points that sit **on the boundary** between the two groups—those that **lie on the outer edges** of the margin street.

Those boundary-touching points are the **support vectors**. They act like **pillars** holding up the edges of the street. If you deleted the other **9,996** points and kept **only** the support vectors, the SVM would place the **same** boundary, because **only those points** constrain the margin.

---

## The math behind the street

Because an SVM draws a **straight line** (or a **flat plane** in 3D), it uses the familiar **linear** form—just written with **vectors**.

The optimization behind that street can be written in two equivalent ways. The **primal** formulation works directly with the geometry above—the weight vector $\mathbf{w}$, the bias $b$, and one constraint per training point. The **dual** formulation, derived later through **Lagrangian multipliers**, rewrites the *same* problem entirely in terms of **pairwise inner products** between training points; that rewrite is what makes the **kernel trick** possible. We start with the primal because it maps cleanly onto the street, then cross over to the dual.

### 1. The equation of the street (primal view)

We're now working in the **primal**: variables are the geometric objects $\mathbf{w}$ and $b$ that *define* the street.

**Center line (decision boundary):**

$$
\mathbf{w} \cdot \mathbf{x} + b = 0
$$

Here $\mathbf{w}$ is the **weight vector** (direction of the boundary), $\mathbf{x}$ is a **data point** in feature space (e.g., power and cadence as coordinates), and $b$ is the **bias** (shifts the boundary).

**The two edges (shoulders of the highway):**

The SVM does not stop at one line—it defines the **two parallel margins** where support vectors can sit.

- **Positive class** (e.g., bonk):  
  $$\mathbf{w} \cdot \mathbf{x} + b = 1$$
- **Negative class** (e.g., no bonk):  
  $$\mathbf{w} \cdot \mathbf{x} + b = -1$$

### 2. Maximizing the margin

Geometrically, the **distance between** the $+1$ and $-1$ hyperplanes is:

$$
\frac{2}{\lVert \mathbf{w} \rVert}
$$

where $\lVert \mathbf{w} \rVert$ is the **length** (Euclidean norm) of $\mathbf{w}$.

So the objective is clear: to make the street **as wide as possible**, maximize $\frac{2}{\lVert \mathbf{w} \rVert}$—equivalently, **minimize** $\lVert \mathbf{w} \rVert$ (subject to constraints that keep the data outside the margin).

### 3. The constraint

The optimizer cannot simply drive $\mathbf{w} \to \mathbf{0}$: a zero weight vector would blow up the width in a meaningless way and violate separation.

Instead, the model **minimizes** $\lVert \mathbf{w} \rVert$ under a **hard rule**: **no training point may fall inside the strip** between the two margins. Every historical example must lie **on or beyond** the correct margin side. In standard binary SVM form (labels $y_i \in \{+1,-1\}$):

$$
y_i \left( \mathbf{w} \cdot \mathbf{x}_i + b \right) \ge 1
$$

**Reading it:** $y_i$ times the signed distance-from-boundary score must be **at least 1**, so each point is **on its class’s margin or farther**—never in the “wrong lane” inside the street.

*(Equivalently, you can write each constraint as $y_i(\mathbf{w} \cdot \mathbf{x}_i + b) - 1 \ge 0$—the form that shows up directly in the Lagrangian below.)*

---

## From primal to dual: The Lagrangian and the kernel bridge

The geometry of the “street” is intuitive; the **optimization** behind it is written as a **primal** problem, then transformed—via **Lagrangian multipliers**—into the **dual**. That step is what makes the **kernel trick** possible.

### 1. The primal problem (what we already know)

The goal is to **maximize the margin**, which is the same as **minimizing** $\lVert \mathbf{w} \rVert$ subject to **no point inside the street**. In optimization it is standard (and mathematically convenient) to minimize the **squared** norm:

$$
\min_{\mathbf{w},\, b} \;\; \frac{1}{2}\lVert \mathbf{w} \rVert^2
$$

**Subject to** a constraint for **every** training example $i = 1,\ldots,n$:

$$
y_i\left(\mathbf{w} \cdot \mathbf{x}_i + b\right) - 1 \ge 0
$$

This is the same feasibility rule as $y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \ge 1$, just rearranged.

### 2. Introducing Lagrangian multipliers

When you must **minimize** a function while staying inside **inequality** constraints, a standard tool is **Lagrangian multipliers**. Instead of treating the objective and the constraints as two separate puzzles, you **fold** the constraints into one expression.

Introduce a multiplier $\alpha_i \ge 0$ for each constraint: it measures how much “pressure” that constraint exerts at the optimum (roughly: penalty for violating the margin).

The **Lagrangian** (primal variables $\mathbf{w}, b$; multipliers $\alpha_1,\ldots,\alpha_n$) is:

$$
L(\mathbf{w}, b, \boldsymbol{\alpha}) =    
\frac{1}{2}\lVert \mathbf{w} \rVert^2
- \sum_{i=1}^{n} \alpha_i \Bigl[ y_i(\mathbf{w} \cdot \mathbf{x}_i + b) - 1 \Bigr]
$$

To move from here toward the dual, you **differentiate** $L$ with respect to $\mathbf{w}$ and $b$, set those derivatives to **zero** (stationarity), solve for $\mathbf{w}$ (and the condition on $b$), and **substitute** back—classic constrained-optimization workflow.

### 3. The punchline: The dual formulation

After imposing stationarity and working through the algebra, the problem can be rewritten **only** in terms of the multipliers $\alpha_i$. The dual (what you actually maximize after flipping signs in some texts) takes the form:

$$
\max_{\boldsymbol{\alpha}} \;\;
\sum_{i=1}^{n} \alpha_i
- \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n}
  \alpha_i \alpha_j \, y_i y_j \, (\mathbf{x}_i \cdot \mathbf{x}_j)
$$

(In full rigor this maximization is subject to **KKT** side conditions—e.g. $\alpha_i \ge 0$ and $\sum_i \alpha_i y_i = 0$ for the separable linear SVM—which pin down feasible $\boldsymbol{\alpha}$.)

**Why this matters for practice**

Look at the **pairwise** term $(\mathbf{x}_i \cdot \mathbf{x}_j)$. In the dual, the algorithm no longer needs to reason from raw coordinates in isolation; it only needs **inner products between pairs** of training points—how each example **relates** to every other.

**Support vectors, from the math:** For most indices $i$, the optimal $\alpha_i$ is **exactly zero**—those points drop out of the sum that builds the solution. The only points with **$\alpha_i > 0$** are the **support vectors**. The algebra is why SVMs can **ignore** almost all of the data and still keep the same boundary.

**The bridge to kernels:** Because the entire dual depends on dot products $\mathbf{x}_i \cdot \mathbf{x}_j$, you can **replace** that inner product with another function $K(\mathbf{x}_i, \mathbf{x}_j)$ that behaves like an inner product in some (possibly **very high–dimensional**) space—without ever materializing those coordinates explicitly. That replacement **is** the **kernel trick**, and the Lagrangian → dual derivation is the step that **exposes** the dot product and makes the swap legitimate.

---

## Part 2: The kernel trick

This is where Support Vector Machines shift from feeling like a **simple geometric rule** (“draw the widest street”) to something **mathematically powerful**: the dual formulation lets them **bend** separation into rich, nonlinear boundaries—**without** paying the full price of explicitly living in gigantic feature spaces.

### The problem: The unsplittable data

We established that an SVM draws a **straight line** (or a **flat plane**) to separate labels. Real data often **won’t cooperate**.

Picture your cycling scatter again: **power** vs **cadence**. Suppose every red **Bonk** ride sits in a tight blob **at the center** of the plot, while every blue **No Bonk** ride forms a **ring** wrapping all the way around that center.

No matter how you swivel a ruler on that 2D page, **one straight cut** cannot separate **inside blob** from **outside ring**. A plain **linear** classifier is ruled out—not because “ML is magic,” but because the **true boundary** isn’t linear in those two axes.

### The solution: Projecting into 3D

If separation fails in 2D, change the geometry you’re reasoning in—**not** necessarily the story you tell about cyclists, but the **coordinate system** where a linear separator becomes possible.

Imagine the scatter printed on **flexible rubber** lying flat on the table. Add a mapping rule tied to geometry: **the closer** a ride is **to the center**, **the farther** its image is lifted **above** the rubber in a new direction (a genuine **third axis**).

- The interior **Bonk** points rise into a sharp **summit**.
- The **No Bonk** ring stays nearer the table.

Now a **flat sheet**—a genuine **affine plane** in 3D—can slide **between** the uplifted reds and the flatter blues. That plane is linear in the **embedded** coordinates: it separates what was **inseparable** along straight lines inside the flat 2D plot.

Perspective matters: viewed back down onto the rubber, the silhouette of where that plane intersects reality can trace out the **curve**—here, intuitively circular—that we needed **all along**. You’ve separated **circle vs center** while still using **linear separation**, but **after** embedding.

![Left: central red disk (“bonk”) surrounded by blue ring (“non-bonk”), not separable by one straight line in 2D; right: lifted coordinates with \(Z = X^2 + Y^2\), separating hyperplane and parallel margins](../images/svm_kernel_projection_2D_to_3D.png)

*Figure:* Lift the scatter with \(Z = f(X,Y) = X^2 + Y^2\) so a **hyperplane** in 3D (with margins) can separate what was inseparable along straight boundaries in Features 1 and 2 alone.

### The trick

Embedded spaces sound great—and then **engineering reality** hits.

If we actually **constructed** lifted coordinates explicitly for millions of riders in 10,000 dimensions, we’d drown CPU and RAM calculating and storing gigantic vectors. Plain matrix code would crater.

Fortunately, the derivation we postponed is quietly doing work: in the dual, training depends on **pairwise comparisons** summarized by **inner products** $\mathbf{x}_i \cdot \mathbf{x}_j$, not “mysteries inside each coordinate taken alone.” That’s precisely what surfaced in the pairwise term $\mathbf{x}_i \cdot \mathbf{x}_j$.

A **kernel**—formally enough, something that induces an inner-product structure—is a computational rule that answers the question:

> *If I implicitly embedded every point into a richer space, what would the inner product between these two images be?*

The **kernel trick** is the shortcut: compute $K(\mathbf{x}_i, \mathbf{x}_j)$ (where K -> Kernel Function) that **agrees** with that hypothetical inner product (dot product), **without** ever materializing the embedded coordinates. The decision boundary you see in the original feature plot can look **wildly nonlinear**, while the algorithm’s heavy lifting still looks like “pairwise kernel evaluations” instead of “allocate the universe.”

### The most common kernels in code

When you fit an SVM in practice, you usually select a kernel—pick the **geometry** in the embedded view that matches the **shape** of your cluster boundaries.

- **Linear kernel:** No embedding trick—just $\mathbf{x}_i \cdot \mathbf{x}_j$. Use when a **flat** separator is appropriate.
- **Polynomial kernel:** Encodes interactions up to some degree—boundaries can flex into **curved** regions (bowls, bends) without hand-engineering every monomial.
- **RBF (Gaussian) kernel:** The famous default for many nonlinear problems. It corresponds to an implicit feature map into an **infinite-dimensional** space (formally, a particular reproducing-kernel Hilbert space). Intuitively, it lets the model place **highly localized** influence around support points and stitch together **flexible** decision regions—at the cost of needing careful **regularization** and hyperparameters so you don’t simply memorize noise.

---

## Takeaway

By **maximizing the margin**, an SVM encourages a **well-generalized** boundary: as much **empty buffer** as possible around the separator, so when **new**, slightly noisy or off-center data arrives, there is **room** to classify it correctly.

**Part 2:** Because the separable dual depends only on **pairwise inner products**, we can swap in a **kernel** $K(\mathbf{x}_i,\mathbf{x}_j)$ and obtain **nonlinear** boundaries in the original feature space while keeping the optimization’s structure—**large margins** in the right space, without explicitly building that space coordinate by coordinate.
