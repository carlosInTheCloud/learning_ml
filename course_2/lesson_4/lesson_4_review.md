# Course 2 — Lesson 4 — Review (questions)

**Answers:** [lesson_4_review_answers.md](lesson_4_review_answers.md)

---

1. In a deep neural network layer with $n^{[l-1]}$ input features and $n^{[l]}$ hidden neurons, what are the dimensions of the weight matrix $\mathbf{W}^{[l]}$ following standard deep learning mathematical conventions?

2. Given an input matrix $\mathbf{X}$ of shape $(n_x, m)$ and a first hidden layer with $n^{[1]}$ neurons, what is the resulting shape of the pre-activation matrix $\mathbf{Z}^{[1]}$ after computing $\mathbf{Z}^{[1]} = \mathbf{W}^{[1]}\mathbf{X} + \mathbf{b}^{[1]}$?

3. During backpropagation, the derivative of the ReLU activation function $g'(z)$ acts as a mathematical “gate.” For a given scalar $z$, what is the value of this derivative if $z < 0$?

4. How is the L2 regularization penalty physically formulated in the cost function $J$ for a neural network with $L$ layers over $m$ examples?

5. When computing backpropagation with L2 regularization, what mathematical term is appended to the standard, unregularized weight gradient equation to calculate $\mathrm{d}\mathbf{W}^{[l]}$?

6. In a standard 2-layer neural network, what is the exact vectorized formulation used to calculate the hidden layer error $\mathrm{d}\mathbf{Z}^{[1]}$ by passing blame backward from $\mathrm{d}\mathbf{Z}^{[2]}$?

7. To compute the output weight gradient $\mathrm{d}\mathbf{W}^{[2]} = \frac{1}{m}\,\mathrm{d}\mathbf{Z}^{[2]}(\text{Input})^{\mathsf T}$, which variable’s transpose strictly belongs in the “Input” placeholder?

8. In vectorized backpropagation across $m$ training examples, how is the bias gradient $\mathrm{d}\mathbf{b}^{[l]}$ calculated from the layer’s error matrix $\mathrm{d}\mathbf{Z}^{[l]}$?

9. What is the mathematical consequence of building a 100-layer neural network but using strictly linear activation functions (i.e. $g(z) = z$) for all hidden layers?

10. For binary classification using binary cross-entropy loss and a sigmoid output activation, the complex derivative of $\mathrm{d}\mathbf{Z}^{[l]}$ simplifies elegantly. What is this simplified form?

11. Given the hyperparameter $\alpha$ (learning rate), which equation correctly represents the execution of the gradient descent update step for a parameter $\mathbf{W}^{[l]}$? (Hint: what’s the new $\mathbf{W}$?)

12. Which NumPy matrix operation accurately executes the vectorized linear step of forward propagation for the first hidden layer across $m$ examples?

13. Why is it standard practice in deep learning to apply L2 regularization penalties strictly to the weight matrices ($\mathbf{W}$) and not to the bias vectors ($\mathbf{b}$)?

14. In a network using ReLU hidden activations, if a specific neuron’s $z$ value is negative for a training example, what happens to its connection weights during that step of the backward pass?

15. The Frobenius norm $\|\mathbf{W}\|_F^2$, used extensively in L2 regularization, is computationally equivalent to which physical operation on the weight matrix?

16. Why is it a catastrophic failure to initialize the weight matrices $\mathbf{W}^{[l]}$ to all zeros in a multi-layer neural network?

17. When executing $\mathbf{Z} = \mathbf{W}\mathbf{X} + \mathbf{b}$ in Python where $\mathbf{W}\mathbf{X}$ is shape $(n, m)$ and $\mathbf{b}$ is shape $(n, 1)$, what specific mechanism allows the single column vector $\mathbf{b}$ to mathematically add to all $m$ columns?

18. When computing the bias gradient via `np.sum(dZ, axis=1, keepdims=True)`, why is the `keepdims=True` argument critical for the code’s stability?

19. Conceptually, when we calculate $\mathrm{d}\mathbf{Z}^{[1]} = (\mathbf{W}^{[2]})^{\mathsf T}\mathrm{d}\mathbf{Z}^{[2]} \odot g'\bigl(\mathbf{Z}^{[1]}\bigr)$, why must we multiply the final error by the transposed weights?

20. During a single training epoch, which values must be “cached” (stored in memory) during forward propagation because they are strictly required to execute backpropagation?
