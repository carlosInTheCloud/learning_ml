# learning_ml

Notes, math, and code labs for **machine learning**—organized from **foundations** (linear algebra and calculus) through **supervised models** (regression, classification, regularization) to **neural networks**.

## How this repo is organized

Material is grouped into **courses** (`course_1`, `course_2`, …). Each course is split into **modules** or **lessons** that are meant to be read **in order**: each unit assumes you are comfortable with the concepts and notation from the earlier ones. Lessons are **stepping stones**—later labs and derivations build directly on earlier definitions (e.g. vectors and matrices before determinants; derivatives before the chain rule and partial derivatives; linear regression before logistic regression; logistic regression and calculus before overfitting and neural nets).

- **Course 1** establishes the **math language** used everywhere later (vectors, matrices, slopes, gradients).
- **Course 2** applies that language to **models**: fit lines and surfaces, classify with logits, control complexity with regularization, stack layers into a small network, then introduce **decision trees** (entropy, information gain) and **ensembles** (bagging / boosting).

Supporting files at the repo root (`requirements.txt`, `setup.sh`) keep the Python environment consistent for labs.

## Syllabus

Read **top to bottom** within each course. Each row is one topic; **click the topic** to open the file (paths are relative to the repo root).

### Course 1 — Mathematics for machine learning

Path: `course_1/`

#### Module 1 — Linear algebra & matrices

Path: `course_1/module_1/`

| Topic |
|-------|
| [Vector space and linear combinations](course_1/module_1/lesson_1_vector_space_and_linear_combinations.md) |
| [Matrix transformations and identity matrices](course_1/module_1/lesson_2_matrix_transformations_and_identity_matrices.md) |
| [Determinants and matrix inversion](course_1/module_1/lesson_3_determinants_and_matrix_inversion.md) |

#### Module 2 — Calculus for machine learning

Path: `course_1/module_2/`

| Topic |
|-------|
| [The derivative](course_1/module_2/lesson_1_the_derivative.md) |
| [The chain rule](course_1/module_2/lesson_2_the_chain_rule.md) |
| [Partial derivatives](course_1/module_2/lesson_3_partial_derivatives.md) |

---

### Course 2 — Supervised learning: regression to neural networks

Path: `course_2/`

#### Lesson 1 — Simple linear regression

Path: `course_2/lesson_1/`

| Topic |
|-------|
| [Simple linear regression](course_2/lesson_1/lesson_1_simple_linear_regression.md) |
| [Simple linear regression lab](course_2/lesson_1/lab_1_linear_regression.md) |

#### Lesson 2 — Logistic regression (binary classification)

Path: `course_2/lesson_2/`

| Topic |
|-------|
| [Logistic regression (binary classification)](course_2/lesson_2/lesson_2_logistic_regression.md) |
| [Logistic regression lab](course_2/lesson_2/lesson_2_logistic_regression_lab.md) |
| [Logistic regression lab (Python)](course_2/lesson_2/lesson_2_logistic_regression_lab.py) |
| [Appendix](course_2/lesson_2/appendix.md) |

#### Lesson 3 — Overfitting, bias–variance, regularization

Path: `course_2/lesson_3/`

| Topic |
|-------|
| [Overfitting, bias–variance tradeoff, and regularization](course_2/lesson_3/overfitting_and_regularization.md) |
| [Overfitting and regularization lab (Python)](course_2/lesson_3/overfitting_and_regularization_lab.py) |

#### Lesson 4 — Neural networks

Path: `course_2/lesson_4/`

| Topic |
|-------|
| [Neural network foundations & forward propagation](course_2/lesson_4/lesson_4a.md) |
| [Backward propagation](course_2/lesson_4/lesson_4b.md) |
| [Manual numerical example](course_2/lesson_4/lesson_4c.md) |
| [Python training loop](course_2/lesson_4/lesson_4d.md) |
| [Lab — train a 2-layer network (NumPy)](course_2/lesson_4/lesson_4_lab.md) |
| [Review — questions](course_2/lesson_4/lesson_4_review.md) |
| [Review — answers](course_2/lesson_4/lesson_4_review_answers.md) |

#### Lesson 5 — Decision trees (entropy, information gain, ensembles)

Path: `course_2/lesson_5/`

| Topic |
|-------|
| [Decision trees: entropy, information gain, CART, pruning & ensembles](course_2/lesson_5/lesson_5.md) |
| [Lab — decision tree from scratch (NumPy)](course_2/lesson_5/lesson_5_lab.md) |
| [Lab — reference implementation (Python)](course_2/lesson_5/lesson_5_lab_answer.py) |
| [Review — questions](course_2/lesson_5/lesson_5_review.md) |
| [Review — answers](course_2/lesson_5/lesson_5_review_answers.md) |

**Assets:** figures for Course 2 live in `course_2/images/`.
