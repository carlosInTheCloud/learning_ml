import csv
from pathlib import Path

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# 1. Generate Non-Linear Data (1,000 rides)
# factor=0.3 creates a distinct gap between the inner and outer circles.
# noise=0.05 adds a little bit of realistic sensor variance.
X_raw, y_raw = make_circles(n_samples=1000, factor=0.3, noise=0.05, random_state=42)

# 2. Split the Data
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")

# 3. Persist the full dataset as a CSV alongside this script.
output_path = Path(__file__).parent / "lesson_6_lab_data.csv"
with output_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Power", "Cadence", "Bonk"])
    for (x0, x1), y in zip(X_raw, y_raw, strict=True):
        writer.writerow([x0, x1, int(y)])

print(f"Wrote {len(X_raw)} rows to {output_path}")