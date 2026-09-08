# 1.1 Linear Algebra

**Part 1 — Mathematical Foundations**

## Scope

The language the rest of the program is written in. A learning problem arrives as a table of numbers, and almost every operation performed on that table — predicting, fitting, projecting, compressing, decomposing — is a statement about matrices. This topic builds those objects and the operations on them, then develops the two structural results that later parts lean on hardest: the eigendecomposition of a symmetric matrix, and the singular value decomposition.

The treatment is a refresher aimed forward. Each subtopic exists because something later needs it, and the connection is stated rather than left implicit.

## Objective

By the end of this topic you should be able to:

- Read a matrix expression and know the shape of every intermediate without computing it.
- Move fluently between the readings of a matrix product — inner products, columns, rows, and sums of outer products — and choose the one that makes a given argument short.
- Decide which of two algebraically equal groupings of a computation to actually perform, and justify it by operation count.
- Derive the eigendecomposition and the SVD, and state precisely what each requires of its input.
- Recognise, in code, the shape and conditioning failures that produce silently wrong answers rather than errors.

## Prerequisites

None. This topic opens the program.

## Syllabus

| | Subtopic | Status |
|---|---|---|
| 1.1.1 | [Vectors and matrices](01_vectors_and_matrices/01_vectors_and_matrices.md) | Core · **available** |
| 1.1.2 | Matrix operations and norms | Core |
| 1.1.3 | Projections and subspaces | Core |
| 1.1.4 | Quadratic forms and positive definiteness | Core |
| 1.1.5 | Eigenvalues and eigenvectors | Core |
| 1.1.6 | Singular value decomposition | Core |
| 1.1.7 | Matrix factorization | Extension |

## Where it is used

- **1.1.1** fixes the design matrix convention used by every subtopic in the program, and introduces the Gram matrix $X^\top X$ that reappears in the normal equations of 4.1.1.
- **1.1.3** makes least squares a projection, which is the geometric content of 4.1.1.
- **1.1.4** supplies the positive semi-definiteness that guarantees the least squares objective has a minimum, and the convexity results of 1.4.1 depend on it.
- **1.1.5** and **1.1.6** are the machinery behind principal component analysis in 7.3.1 and 7.3.2.
