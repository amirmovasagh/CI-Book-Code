# Example 1.8: Splitting the data into training, validation, and test sets (60/20/20)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
 
X, y = load_breast_cancer(return_X_y=True)
 
# Step 1: set aside 20% for testing
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)
 
# Step 2: from the remaining 80%, one quarter (= 20% of the total) for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
 
n = len(X)
for name, part in [("Train", X_train), ("Validation", X_val), ("Test", X_test)]:
    print(f"{name:<11}: {len(part):>3} samples ({len(part) / n:.0%})")
print("Benign ratio in train/val/test:",
      round(y_train.mean(), 3), round(y_val.mean(), 3), round(y_test.mean(), 3))
