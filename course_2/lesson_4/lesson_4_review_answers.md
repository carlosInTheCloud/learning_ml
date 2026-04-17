# Course 2 — Lesson 4 — Review (answers)

**Questions:** [lesson_4_review.md](lesson_4_review.md)

---

1. **$(n^{[l]},\, n^{[l-1]})$**. The weight matrix is **(neurons, inputs)**—rows correspond to neurons in layer $l$, columns to inputs from layer $l-1$.

2. Resulting shape prior to activation: dimensions line up as **$(n^{[1]},\, n_x) \times (n_x,\, m) \rightarrow (n^{[1]},\, m)$**. That matrix holds the **score of every neuron across every training sample**.

3. **$0$**. ReLU outputs $0$ if $z < 0$ and $z$ if $z > 0$; the derivative for $z < 0$ is **$0$**.

4. Add a penalty proportional to the **sum of squared Frobenius norms** of the weight matrices, e.g. **$\dfrac{\lambda}{2m} \displaystyle\sum_{\ell=1}^{L} \|\mathbf{W}^{[\ell]}\|_F^2$**. The **double bars** denote a **norm** (here the **Frobenius** norm of the matrix); the **subscript $F$** means “Frobenius.” **$\|\mathbf{W}\|_F^2$** is the sum of the squares of all entries of $\mathbf{W}$.

5. **$\dfrac{\lambda}{m}\,\mathbf{W}^{[l]}$**. When you differentiate **$\dfrac{\lambda}{2m}\|\mathbf{W}^{[l]}\|_F^2$** with respect to $\mathbf{W}^{[l]}$, the power of $2$ brings down a factor that **cancels the $2$** in **$\lambda/(2m)$**, leaving **$\lambda/m$** times $\mathbf{W}^{[l]}$.

6. **$\mathrm{d}\mathbf{Z}^{[1]} = (\mathbf{W}^{[2]})^{\mathsf T}\,\mathrm{d}\mathbf{Z}^{[2]} \odot g'\bigl(\mathbf{Z}^{[1]}\bigr)$** (element-wise product with the ReLU derivative mask).

7. **$\mathbf{A}^{[1]}$** (the activations fed into layer 2).

8. **$\mathrm{d}\mathbf{b}^{[l]} = \dfrac{1}{m}\,\mathrm{d}\mathbf{Z}^{[l]}$** in the “sum over examples, then divide by $m$” sense—implemented as **`(1/m) * np.sum(dZ, axis=1, keepdims=True)`** when examples are columns.

9. The network **collapses** into a **single linear function** end-to-end, so depth buys **no extra expressive power** beyond one linear layer.

10. **$\mathrm{d}\mathbf{Z}^{[l]} = \mathbf{A}^{[l]} - \mathbf{Y}$** at the sigmoid output (with the usual BCE setup).

11. First compute the gradient (e.g. **$\mathrm{d}\mathbf{W}^{[l]} = \dfrac{1}{m}\,\mathrm{d}\mathbf{Z}^{[l+1]}(\mathbf{A}^{[l]})^{\mathsf T}$** for the appropriate layer, plus L2 if used). Then update: **$\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \alpha\,\mathrm{d}\mathbf{W}^{[l]}$**.

12. **`Z1 = np.dot(W1, X) + b1`** (with shapes chosen so **`W1`** is **`(n^{[1]}, n_x)`** and **`X`** is **`(n_x, m)`**).

13. The **bias** doesn’t change the **shape** of the decision boundary in the same way as weights—it **shifts** it. **Overfitting** is usually tied to **large weights**; biases are often left **unregularized**.

14. The **error gate is closed** (**$0$** derivative through ReLU), so **no gradient** from the loss flows through that neuron on that example for that path—it **learns nothing** from that data point along those connections.

15. **Square every entry** of $\mathbf{W}$ and **add them all up**—that sum is **$\|\mathbf{W}\|_F^2$** (squared Frobenius norm).

16. **Symmetry is not broken**: neurons in a layer can compute the **same** thing, get the **same** gradients, and **learn the same features**—the network fails to diversify.

17. **Broadcasting**: NumPy **replicates** $\mathbf{b}$ along the $m$ dimension so it **adds** to every column of $\mathbf{W}\mathbf{X}$.

18. Without **`keepdims=True`**, **`np.sum`** can return a **1D array** of shape **`(n,)`** instead of **`(n, 1)`**, which **breaks broadcasting** when you subtract or add next to **`(n, m)$`** tensors. **`keepdims=True`** preserves the **$(n, 1)$** shape so dimensions stay consistent.

19. **$(\mathbf{W}^{[2]})^{\mathsf T}$** maps the error to the **same dimension** as layer 1’s pre-activations and **routes blame** according to how much layer 2 **relied on each** hidden unit.

20. **$\mathbf{Z}$** and **$\mathbf{A}$** (per layer, as needed for derivatives and for inputs to the next layer backward).
