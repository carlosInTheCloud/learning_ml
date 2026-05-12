#################################################################################
# Setup the training data
#################################################################################
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

print("Setup the training data...")
# 1_000 rides, inner vs outer circle (factor & noise match the data generator lab script)
X_raw, y_raw = make_circles(
    n_samples=1000, factor=0.3, noise=0.05, random_state=42
)

# 80 % train / 20 % test → 800 training rows, 200 test rows
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)
print("Training data setup complete")
#################################################################################
# Task 1: Prove the primal limitation
#################################################################################
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

primal_svm = LinearSVC(loss='hinge', dual=False, C=1.0)
primal_svm.fit(X_train, y_train)

print(f"Task 1: Primal SVM accuracy: {accuracy_score(y_test, primal_svm.predict(X_test))}")

#################################################################################
# Task 2: Prove the dual + kernel path
#################################################################################
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

dual_svm = SVC(kernel='rbf', C=1.0)
dual_svm.fit(X_train, y_train)

print(f"Task 2: Dual + kernel SVM accuracy: {accuracy_score(y_test, dual_svm.predict(X_test))}")

#################################################################################
# Task 3: Verify sparsity
#################################################################################
n_sv = dual_svm.support_vectors_.shape[0]
inactive = len(X_train) - n_sv
print(f"Task 3: Support vectors {n_sv}, inactive {inactive}")

#################################################################################
# Task 4: Test the box constraint (`C`)
#################################################################################
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

sparse_svm = SVC(kernel='rbf', C=0.001)
sparse_svm.fit(X_train, y_train)

print(f"Task 4: Sparse SVM accuracy: {accuracy_score(y_test, sparse_svm.predict(X_test))}")
n_sv = sparse_svm.support_vectors_.shape[0]
inactive = len(X_train) - n_sv
print(f"Task 4: Support vectors {n_sv}, inactive {inactive}")