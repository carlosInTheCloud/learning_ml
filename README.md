# Learning ML

A from-scratch, graduate-level machine learning curriculum, calibrated to Stanford's CS229 / CS230 / CS231n / CS224n.
Every model is derived from first principles, implemented in NumPy, and verified numerically.

The standing rules for this repository live in [`OKF/`](OKF/):

- [goal.md](OKF/goal.md) — the objective, the standard, and what counts as mastery
- [overall_program.md](OKF/overall_program.md) — the full curriculum
- [agent_teaching_method.md](OKF/agent_teaching_method.md) — how material is structured and authored
- [conventions.md](OKF/conventions.md) — notation, math rendering, and code standards
- [postponed.md](OKF/postponed.md) — what is deliberately out of scope, and why

---

## Progress

**0 / 143 core** subtopics complete &nbsp;·&nbsp; **0 / 35 extension** &nbsp;·&nbsp; 178 total

Unmarked entries are **core** — the critical path. Entries marked *ext* are safely deferrable on a first pass;
nothing later in the program derives from them.

### 1. Mathematical Foundations

`25 core` &nbsp; `4 extension`

**1.1 Linear Algebra**

- [ ] 1.1.1 Vectors and matrices
- [ ] 1.1.2 Matrix operations and norms
- [ ] 1.1.3 Projections and subspaces
- [ ] 1.1.4 Quadratic forms and positive definiteness
- [ ] 1.1.5 Eigenvalues and eigenvectors
- [ ] 1.1.6 Singular value decomposition
- [ ] 1.1.7 Matrix factorization *(ext)*

**1.2 Calculus**

- [ ] 1.2.1 Partial derivatives
- [ ] 1.2.2 Gradients and directional derivatives
- [ ] 1.2.3 Chain rule
- [ ] 1.2.4 Matrix and vector calculus
- [ ] 1.2.5 Taylor expansion and the Hessian
- [ ] 1.2.6 Optimization landscapes, convexity, and critical points

**1.3 Probability and Statistics**

- [ ] 1.3.1 Random variables
- [ ] 1.3.2 Probability distributions
- [ ] 1.3.3 Expectation and variance
- [ ] 1.3.4 Joint, marginal, and conditional distributions
- [ ] 1.3.5 Bayes' rule
- [ ] 1.3.6 Maximum likelihood estimation
- [ ] 1.3.7 Entropy and KL divergence
- [ ] 1.3.8 Sampling distributions and the central limit theorem *(ext)*
- [ ] 1.3.9 Hypothesis testing *(ext)*
- [ ] 1.3.10 Confidence intervals *(ext)*

**1.4 Optimization**

- [ ] 1.4.1 Convex sets and convex functions
- [ ] 1.4.2 Gradient descent
- [ ] 1.4.3 Stochastic and mini-batch gradient descent
- [ ] 1.4.4 Newton's method and second-order optimization
- [ ] 1.4.5 Constrained optimization and Lagrange multipliers
- [ ] 1.4.6 KKT conditions

### 2. Core Supervised Learning Concepts

`16 core` &nbsp; `1 extension`

**2.1 Learning Framework**

- [ ] 2.1.1 Inputs/features vs outputs/labels
- [ ] 2.1.2 Hypothesis classes and empirical risk
- [ ] 2.1.3 Training/validation/test splits
- [ ] 2.1.4 Cross-validation
- [ ] 2.1.5 Generalization
- [ ] 2.1.6 Overfitting vs underfitting
- [ ] 2.1.7 Bias-variance tradeoff

**2.2 Loss Functions**

- [ ] 2.2.1 Loss functions from maximum likelihood
- [ ] 2.2.2 Mean squared error
- [ ] 2.2.3 Cross-entropy loss
- [ ] 2.2.4 Hinge loss

**2.3 Evaluation Metrics**

- [ ] 2.3.1 Accuracy and the confusion matrix
- [ ] 2.3.2 Precision and recall
- [ ] 2.3.3 F1 score
- [ ] 2.3.4 ROC-AUC and precision-recall curves
- [ ] 2.3.5 RMSE/MAE
- [ ] 2.3.6 Calibration *(ext)*

### 3. Data Processing and Feature Engineering

`8 core` &nbsp; `4 extension`

**3.1 Data Preparation**

- [ ] 3.1.1 Data types, schemas, and exploratory analysis
- [ ] 3.1.2 Missing data handling
- [ ] 3.1.3 Outliers and robust statistics *(ext)*
- [ ] 3.1.4 Encoding categorical variables
- [ ] 3.1.5 Normalization and standardization

**3.2 Feature Construction**

- [ ] 3.2.1 Interaction terms
- [ ] 3.2.2 Basis expansion and polynomial features
- [ ] 3.2.3 Binning and discretization *(ext)*
- [ ] 3.2.4 Aggregate and temporal features *(ext)*

**3.3 Data Integrity**

- [ ] 3.3.1 Data leakage
- [ ] 3.3.2 Train/test contamination
- [ ] 3.3.3 Distribution shift *(ext)*

### 4. Regression Methods

`12 core` &nbsp; `2 extension`

**4.1 Linear Regression**

- [ ] 4.1.1 Ordinary least squares and the normal equations
- [ ] 4.1.2 Least squares as maximum likelihood
- [ ] 4.1.3 Gradient descent for linear regression
- [ ] 4.1.4 Feature scaling and conditioning
- [ ] 4.1.5 Polynomial regression
- [ ] 4.1.6 Multicollinearity

**4.2 Regularized Regression**

- [ ] 4.2.1 Regularization and the bias-variance tradeoff
- [ ] 4.2.2 Ridge regression
- [ ] 4.2.3 Lasso
- [ ] 4.2.4 Elastic net *(ext)*
- [ ] 4.2.5 Selecting the regularization strength

**4.3 Generalized Linear Models**

- [ ] 4.3.1 The exponential family
- [ ] 4.3.2 GLM construction and link functions
- [ ] 4.3.3 Poisson regression *(ext)*

### 5. Classification Algorithms

`18 core` &nbsp; `2 extension`

**5.1 Logistic Regression**

- [ ] 5.1.1 Binary classification and the logistic model
- [ ] 5.1.2 Maximum likelihood and cross-entropy
- [ ] 5.1.3 Gradient descent and Newton's method (IRLS)
- [ ] 5.1.4 Multiclass classification
- [ ] 5.1.5 Softmax regression

**5.2 Support Vector Machines**

- [ ] 5.2.1 Maximum margin classifiers
- [ ] 5.2.2 The Lagrangian dual and KKT conditions
- [ ] 5.2.3 Kernels and the kernel trick
- [ ] 5.2.4 Soft-margin SVMs
- [ ] 5.2.5 Sequential minimal optimization *(ext)*

**5.3 Decision Trees**

- [ ] 5.3.1 Entropy
- [ ] 5.3.2 Information gain
- [ ] 5.3.3 Gini impurity
- [ ] 5.3.4 CART and splitting criteria
- [ ] 5.3.5 Tree pruning

**5.4 Ensemble Methods**

- [ ] 5.4.1 Bagging and the bias-variance effect
- [ ] 5.4.2 Random forests
- [ ] 5.4.3 AdaBoost
- [ ] 5.4.4 Gradient boosting
- [ ] 5.4.5 XGBoost *(ext)*

### 6. Probabilistic and Bayesian Learning

`6 core` &nbsp; `2 extension`

**6.1 Generative Classifiers**

- [ ] 6.1.1 Generative vs discriminative models
- [ ] 6.1.2 Gaussian discriminant analysis
- [ ] 6.1.3 Naive Bayes
- [ ] 6.1.4 Laplace smoothing *(ext)*

**6.2 Bayesian Methods**

- [ ] 6.2.1 Priors, likelihoods, and posteriors
- [ ] 6.2.2 MAP estimation
- [ ] 6.2.3 Bayesian linear regression
- [ ] 6.2.4 Gaussian processes *(ext)*

### 7. Unsupervised Learning

`14 core` &nbsp; `5 extension`

**7.1 Clustering**

- [ ] 7.1.1 k-means and Lloyd's algorithm
- [ ] 7.1.2 k-means++ and convergence properties
- [ ] 7.1.3 Hierarchical clustering
- [ ] 7.1.4 Density-based clustering (DBSCAN)
- [ ] 7.1.5 Evaluating clusterings

**7.2 Latent Variable Models**

- [ ] 7.2.1 Gaussian mixture models
- [ ] 7.2.2 The EM algorithm
- [ ] 7.2.3 The evidence lower bound and convergence
- [ ] 7.2.4 Factor analysis *(ext)*
- [ ] 7.2.5 Matrix factorization for collaborative filtering

**7.3 Dimensionality Reduction**

- [ ] 7.3.1 Principal component analysis
- [ ] 7.3.2 PCA via SVD
- [ ] 7.3.3 Kernel PCA *(ext)*
- [ ] 7.3.4 Linear discriminant analysis
- [ ] 7.3.5 Independent component analysis *(ext)*
- [ ] 7.3.6 t-SNE
- [ ] 7.3.7 UMAP *(ext)*

**7.4 Density Estimation and Anomaly Detection**

- [ ] 7.4.1 Kernel density estimation *(ext)*
- [ ] 7.4.2 Anomaly detection

### 8. Statistical Learning Theory

`6 core` &nbsp; `0 extension`

**8.1 Foundations**

- [ ] 8.1.1 Empirical risk minimization
- [ ] 8.1.2 PAC learning
- [ ] 8.1.3 VC dimension
- [ ] 8.1.4 Generalization bounds

**8.2 Regularized Objectives**

- [ ] 8.2.1 Structural risk minimization
- [ ] 8.2.2 The bias-variance decomposition revisited

### 9. Neural Networks and Deep Learning

`30 core` &nbsp; `10 extension`

**9.1 Perceptrons and MLPs**

- [ ] 9.1.1 The perceptron
- [ ] 9.1.2 Feedforward networks
- [ ] 9.1.3 Activation functions
- [ ] 9.1.4 Universal approximation *(ext)*
- [ ] 9.1.5 Backpropagation
- [ ] 9.1.6 Computational graphs and automatic differentiation

**9.2 Training Deep Networks**

- [ ] 9.2.1 Weight initialization
- [ ] 9.2.2 Vanishing and exploding gradients
- [ ] 9.2.3 Batch normalization
- [ ] 9.2.4 Layer normalization and residual connections
- [ ] 9.2.5 Dropout and regularization
- [ ] 9.2.6 Momentum, RMSProp, and Adam
- [ ] 9.2.7 Learning rate schedules *(ext)*

**9.3 Convolutional Networks**

- [ ] 9.3.1 The convolution operation
- [ ] 9.3.2 Pooling and stride
- [ ] 9.3.3 CNN architectures
- [ ] 9.3.4 Data augmentation *(ext)*
- [ ] 9.3.5 Transfer learning *(ext)*
- [ ] 9.3.6 Object detection and segmentation *(ext)*

**9.4 Sequence Models**

- [ ] 9.4.1 Recurrent neural networks
- [ ] 9.4.2 Backpropagation through time
- [ ] 9.4.3 LSTMs and GRUs
- [ ] 9.4.4 Word embeddings
- [ ] 9.4.5 Sequence-to-sequence models

**9.5 Attention and Transformers**

- [ ] 9.5.1 Attention mechanisms
- [ ] 9.5.2 Self-attention and multi-head attention
- [ ] 9.5.3 Positional encoding
- [ ] 9.5.4 The transformer block
- [ ] 9.5.5 Pretraining and fine-tuning
- [ ] 9.5.6 Large language models *(ext)*

**9.6 Representation Learning and Retrieval**

- [ ] 9.6.1 Embedding spaces and similarity
- [ ] 9.6.2 Metric learning and triplet loss
- [ ] 9.6.3 Contrastive learning and InfoNCE
- [ ] 9.6.4 Negative sampling and in-batch negatives
- [ ] 9.6.5 Two-tower (dual-encoder) retrieval
- [ ] 9.6.6 Approximate nearest neighbor search *(ext)*

**9.7 Deep Generative Models**

- [ ] 9.7.1 Autoencoders
- [ ] 9.7.2 Variational autoencoders *(ext)*
- [ ] 9.7.3 Generative adversarial networks *(ext)*
- [ ] 9.7.4 Diffusion models *(ext)*

### 10. Practical ML Systems

`4 core` &nbsp; `4 extension`

**10.1 Experimentation**

- [ ] 10.1.1 ML pipelines
- [ ] 10.1.2 Hyperparameter tuning
- [ ] 10.1.3 Experiment tracking *(ext)*

**10.2 Deployment**

- [ ] 10.2.1 Model deployment *(ext)*
- [ ] 10.2.2 Monitoring and retraining *(ext)*

**10.3 Frameworks**

- [ ] 10.3.1 scikit-learn
- [ ] 10.3.2 PyTorch
- [ ] 10.3.3 TensorFlow/JAX *(ext)*

### 11. Ethics and Responsible AI

`4 core` &nbsp; `1 extension`

**11.1 Fairness and Bias**

- [ ] 11.1.1 Fairness criteria
- [ ] 11.1.2 Bias in datasets
- [ ] 11.1.3 Privacy *(ext)*

**11.2 Interpretability and Safety**

- [ ] 11.2.1 Explainability and model interpretability
- [ ] 11.2.2 Safety and robustness
