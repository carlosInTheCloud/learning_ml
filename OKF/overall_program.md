---
type: Curriculum
title: Overall Program
description: Full curriculum outline spanning mathematical foundations, data processing, supervised and unsupervised learning, learning theory, deep learning, and responsible AI.
tags: [curriculum, syllabus, roadmap]
status: stable
generated: { by: human:carlos.espinosa, at: 2026-09-07T15:16:33Z }
trigger: always_on
---

# Overall Program

Structure follows [agent_teaching_method.md](agent_teaching_method.md): top-level entries are **parts**, their children are **topics**, and the leaves are **subtopics** — the unit of instruction. Content is developed in this order.

Subtopics marked ***extension*** are safely deferrable on a first pass: nothing later in the program derives from them. Everything unmarked is **core** — the critical path. An extension is postponed, never skipped; it stays in place beside its siblings so the topic reads whole.


1. **Mathematical Foundations**
Prerequisites for everything that follows. Covered only where later parts consume them, but covered to full depth where they do.
   1. Linear Algebra
      1. Vectors and matrices
      2. Matrix operations and norms
      3. Projections and subspaces
      4. Quadratic forms and positive definiteness
      5. Eigenvalues and eigenvectors
      6. Singular value decomposition
      7. Matrix factorization — *extension*
   2. Calculus
      1. Partial derivatives
      2. Gradients and directional derivatives
      3. Chain rule
      4. Matrix and vector calculus
      5. Taylor expansion and the Hessian
      6. Optimization landscapes, convexity, and critical points
   3. Probability and Statistics
      1. Random variables
      2. Probability distributions
      3. Expectation and variance
      4. Joint, marginal, and conditional distributions
      5. Bayes' rule
      6. Maximum likelihood estimation
      7. Entropy and KL divergence
      8. Sampling distributions and the central limit theorem — *extension*
      9. Hypothesis testing — *extension*
      10. Confidence intervals — *extension*
   4. Optimization
      1. Convex sets and convex functions
      2. Gradient descent
      3. Stochastic and mini-batch gradient descent
      4. Newton's method and second-order optimization
      5. Constrained optimization and Lagrange multipliers
      6. KKT conditions
2. **Core Supervised Learning Concepts**
   1. Learning Framework
      1. Inputs/features vs outputs/labels
      2. Hypothesis classes and empirical risk
      3. Training/validation/test splits
      4. Cross-validation
      5. Generalization
      6. Overfitting vs underfitting
      7. Bias-variance tradeoff
   2. Loss Functions
      1. Loss functions from maximum likelihood
      2. Mean squared error
      3. Cross-entropy loss
      4. Hinge loss
   3. Evaluation Metrics
      1. Accuracy and the confusion matrix
      2. Precision and recall
      3. F1 score
      4. ROC-AUC and precision-recall curves
      5. RMSE/MAE
      6. Calibration — *extension*
3. **Data Processing and Feature Engineering**
Placed before modeling because this is where the modeling cycle actually begins, and returned to throughout.
   1. Data Preparation
      1. Data types, schemas, and exploratory analysis
      2. Missing data handling
      3. Outliers and robust statistics — *extension*
      4. Encoding categorical variables
      5. Normalization and standardization
   2. Feature Construction
      1. Interaction terms
      2. Basis expansion and polynomial features
      3. Binning and discretization — *extension*
      4. Aggregate and temporal features — *extension*
   3. Data Integrity
      1. Data leakage
      2. Train/test contamination
      3. Distribution shift — *extension*
4. **Regression Methods**
   1. Linear Regression
      1. Ordinary least squares and the normal equations
      2. Least squares as maximum likelihood
      3. Gradient descent for linear regression
      4. Feature scaling and conditioning
      5. Polynomial regression
      6. Multicollinearity
   2. Regularized Regression
      1. Regularization and the bias-variance tradeoff
      2. Ridge regression
      3. Lasso
      4. Elastic net — *extension*
      5. Selecting the regularization strength
   3. Generalized Linear Models
      1. The exponential family
      2. GLM construction and link functions
      3. Poisson regression — *extension*
5. **Classification Algorithms**
   1. Logistic Regression
      1. Binary classification and the logistic model
      2. Maximum likelihood and cross-entropy
      3. Gradient descent and Newton's method (IRLS)
      4. Multiclass classification
      5. Softmax regression
   2. Support Vector Machines
      1. Maximum margin classifiers
      2. The Lagrangian dual and KKT conditions
      3. Kernels and the kernel trick
      4. Soft-margin SVMs
      5. Sequential minimal optimization — *extension*
   3. Decision Trees
      1. Entropy
      2. Information gain
      3. Gini impurity
      4. CART and splitting criteria
      5. Tree pruning
   4. Ensemble Methods
      1. Bagging and the bias-variance effect
      2. Random forests
      3. AdaBoost
      4. Gradient boosting
      5. XGBoost — *extension*
6. **Probabilistic and Bayesian Learning**
   1. Generative Classifiers
      1. Generative vs discriminative models
      2. Gaussian discriminant analysis
      3. Naive Bayes
      4. Laplace smoothing — *extension*
   2. Bayesian Methods
      1. Priors, likelihoods, and posteriors
      2. MAP estimation
      3. Bayesian linear regression
      4. Gaussian processes — *extension*
7. **Unsupervised Learning**
   1. Clustering
      1. k-means and Lloyd's algorithm
      2. k-means++ and convergence properties
      3. Hierarchical clustering
      4. Density-based clustering (DBSCAN)
      5. Evaluating clusterings
   2. Latent Variable Models
      1. Gaussian mixture models
      2. The EM algorithm
      3. The evidence lower bound and convergence
      4. Factor analysis — *extension*
      5. Matrix factorization for collaborative filtering
   3. Dimensionality Reduction
      1. Principal component analysis
      2. PCA via SVD
      3. Kernel PCA — *extension*
      4. Linear discriminant analysis
      5. Independent component analysis — *extension*
      6. t-SNE
      7. UMAP — *extension*
   4. Density Estimation and Anomaly Detection
      1. Kernel density estimation — *extension*
      2. Anomaly detection
8. **Statistical Learning Theory**
   1. Foundations
      1. Empirical risk minimization
      2. PAC learning
      3. VC dimension
      4. Generalization bounds
   2. Regularized Objectives
      1. Structural risk minimization
      2. The bias-variance decomposition revisited
9. **Neural Networks and Deep Learning**
   1. Perceptrons and MLPs
      1. The perceptron
      2. Feedforward networks
      3. Activation functions
      4. Universal approximation — *extension*
      5. Backpropagation
      6. Computational graphs and automatic differentiation
   2. Training Deep Networks
      1. Weight initialization
      2. Vanishing and exploding gradients
      3. Batch normalization
      4. Layer normalization and residual connections
      5. Dropout and regularization
      6. Momentum, RMSProp, and Adam
      7. Learning rate schedules — *extension*
   3. Convolutional Networks
      1. The convolution operation
      2. Pooling and stride
      3. CNN architectures
      4. Data augmentation — *extension*
      5. Transfer learning — *extension*
      6. Object detection and segmentation — *extension*
   4. Sequence Models
      1. Recurrent neural networks
      2. Backpropagation through time
      3. LSTMs and GRUs
      4. Word embeddings
      5. Sequence-to-sequence models
   5. Attention and Transformers
      1. Attention mechanisms
      2. Self-attention and multi-head attention
      3. Positional encoding
      4. The transformer block
      5. Pretraining and fine-tuning
      6. Large language models — *extension*
   6. Representation Learning and Retrieval
      1. Embedding spaces and similarity
      2. Metric learning and triplet loss
      3. Contrastive learning and InfoNCE
      4. Negative sampling and in-batch negatives
      5. Two-tower (dual-encoder) retrieval
      6. Approximate nearest neighbor search — *extension*
   7. Deep Generative Models
      1. Autoencoders
      2. Variational autoencoders — *extension*
      3. Generative adversarial networks — *extension*
      4. Diffusion models — *extension*
10. **Practical ML Systems**
   1. Experimentation
      1. ML pipelines
      2. Hyperparameter tuning
      3. Experiment tracking — *extension*
   2. Deployment
      1. Model deployment — *extension*
      2. Monitoring and retraining — *extension*
   3. Frameworks
      1. scikit-learn
      2. PyTorch
      3. TensorFlow/JAX — *extension*
11. **Ethics and Responsible AI**
   1. Fairness and Bias
      1. Fairness criteria
      2. Bias in datasets
      3. Privacy — *extension*
   2. Interpretability and Safety
      1. Explainability and model interpretability
      2. Safety and robustness
