import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb

# ==========================================
# PHASE 1: The Data Vault
# ==========================================
print("Generating 10,000 Historical Cycling Rides...")
# Simulating a dataset where ~20% of rides result in a Bonk (class 1)
X_raw, y_raw = make_classification(
    n_samples=10000, 
    n_features=10, 
    n_informative=6, 
    weights=[0.8, 0.2], # 80% No Bonk, 20% Bonk
    random_state=42
)

# Locking 20% of the data in the test vault
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

# ==========================================
# PHASE 2: The Parallel Factory (Random Forest)
# ==========================================
print("\n[1/2] Initiating Random Forest Grid Search...")
rf_start_time = time.time()

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_grid = {
    'n_estimators': [100, 300],         # Coarse search for the plateau
    'max_depth': [None, 10],            # Pruning
    'max_features': ['sqrt', 'log2']    # Blindfolding
}

rf_tuner = GridSearchCV(rf_base, rf_grid, cv=3, scoring='accuracy')
rf_tuner.fit(X_train, y_train)
rf_champion = rf_tuner.best_estimator_

print(f"--> RF Search Completed in {time.time() - rf_start_time:.2f} seconds.")
print(f"--> Winning Factory Settings: {rf_tuner.best_params_}")

# ==========================================
# PHASE 3: The Sequential Chain (XGBoost)
# ==========================================
print("\n[2/2] Initiating XGBoost Grid Search...")
xgb_start_time = time.time()

xgb_base = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
xgb_grid = {
    'n_estimators': [100, 300],         # Length of the chain
    'learning_rate': [0.01, 0.1],       # Size of the calculus step
    'max_depth': [3, 5]                 # Keep trees shallow
}

xgb_tuner = GridSearchCV(xgb_base, xgb_grid, cv=3, scoring='accuracy')
xgb_tuner.fit(X_train, y_train)
xgb_champion = xgb_tuner.best_estimator_

print(f"--> XGBoost Search Completed in {time.time() - xgb_start_time:.2f} seconds.")
print(f"--> Winning Chain Settings: {xgb_tuner.best_params_}")

# ==========================================
# PHASE 4: The Final Exam (Hold-Out Validation)
# ==========================================
print("\n==========================================")
print("FINAL EVALUATION: RUNNING INFERENCE ON TEST VAULT")
print("==========================================\n")

rf_predictions = rf_champion.predict(X_test)
xgb_predictions = xgb_champion.predict(X_test)

print(f"Random Forest Accuracy:  {accuracy_score(y_test, rf_predictions) * 100:.2f}%")
print(f"XGBoost Accuracy:        {accuracy_score(y_test, xgb_predictions) * 100:.2f}%")

print("\n--- RANDOM FOREST CONFUSION MATRIX ---")
print("                  [Predicted Normal]  [Predicted Bonk]")
cm_rf = confusion_matrix(y_test, rf_predictions)
print(f"[Actual Normal]   {cm_rf[0][0]:<19} {cm_rf[0][1]}")
print(f"[Actual Bonk]     {cm_rf[1][0]:<19} {cm_rf[1][1]}")

print("\n--- RANDOM FOREST BUSINESS REPORT ---")
print(classification_report(y_test, rf_predictions, target_names=["Normal Ride", "Bonk Ride"]))

print("\n--- XGBOOST CONFUSION MATRIX ---")
print("                  [Predicted Normal]  [Predicted Bonk]")
cm_xgb = confusion_matrix(y_test, xgb_predictions)
print(f"[Actual Normal]   {cm_xgb[0][0]:<19} {cm_xgb[0][1]}")
print(f"[Actual Bonk]     {cm_xgb[1][0]:<19} {cm_xgb[1][1]}")

print("\n--- XGBOOST BUSINESS REPORT ---")
print(classification_report(y_test, xgb_predictions, target_names=["Normal Ride", "Bonk Ride"]))