# logistic_regression.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Helper functions
# -------------------------
def sigmoid(z):
    """Sigmoid / logistic function"""
    return 1.0 / (1.0 + np.exp(-z))

def compute_cost(X, y, w, b):
    """Binary cross-entropy cost"""
    m = len(y)
    z = X.dot(w) + b
    y_pred = sigmoid(z)
    # clip to avoid log(0)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    cost = - (1.0 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return cost

def gradients(X, y, w, b):
    """Return gradients dw and db"""
    m = len(y)
    z = X.dot(w) + b
    y_pred = sigmoid(z)
    error = y_pred - y
    dw = (1.0 / m) * X.T.dot(error)        # shape: (n_features, 1)
    db = (1.0 / m) * np.sum(error)        # scalar
    return dw, db

def predict_proba(X, w, b):
    """Return predicted probabilities"""
    return sigmoid(X.dot(w) + b)

def predict(X, w, b, threshold=0.5):
    """Return binary predictions (0/1)"""
    probs = predict_proba(X, w, b)
    return (probs >= threshold).astype(int)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# -------------------------
# Load and prepare data
# -------------------------
def load_data(path="data.csv"):
    df = pd.read_csv(path)
    X = df[["hours"]].values.astype(float)   # shape (m, 1)
    y = df["passed"].values.reshape(-1, 1).astype(float)  # shape (m, 1)
    return X, y

# Feature scaling (important for gradient descent stability on real datasets)
def scale_features(X):
    """Simple min-max scaling to [0,1]"""
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_scaled = (X - X_min) / (X_max - X_min + 1e-12)
    return X_scaled, X_min, X_max

# -------------------------
# Training function
# -------------------------
def train(X, y, learning_rate=0.1, epochs=1000, verbose=True):
    m, n = X.shape
    w = np.zeros((n, 1))
    b = 0.0
    costs = []

    for i in range(epochs):
        dw, db = gradients(X, y, w, b)

        # update
        w -= learning_rate * dw
        b -= learning_rate * db

        if i % 50 == 0 or i == epochs - 1:
            cost = compute_cost(X, y, w, b)
            costs.append((i, cost))
            if verbose:
                print(f"Epoch {i:4d} | Cost: {cost:.6f}")

    return w, b, costs

# -------------------------
# Main: run training and evaluation
# -------------------------
if __name__ == "__main__":
    # 1. Load
    X_raw, y = load_data("data.csv")

    # 2. Scale features
    X, X_min, X_max = scale_features(X_raw)

    # 3. Train
    print("Training logistic regression (from scratch)...")
    w, b, costs = train(X, y, learning_rate=0.5, epochs=1000)

    # 4. Evaluate on training data
    y_pred = predict(X, w, b)
    acc = accuracy(y, y_pred)
    print(f"\nTraining accuracy: {acc*100:.2f}%")
    print(f"Weights: {w.ravel()}")
    print(f"Bias: {b}")

    # 5. Show training cost trend (optional)
    epochs_list, cost_list = zip(*costs)
    plt.plot(epochs_list, cost_list)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("Training cost vs Epoch")
    plt.grid(True)
    plt.show()

    # 6. Try predictions for new examples
    test_hours = np.array([[2], [5], [8]], dtype=float)
    # scale them using same min/max
    test_scaled = (test_hours - X_min) / (X_max - X_min + 1e-12)
    probs = predict_proba(test_scaled, w, b)
    preds = predict(test_scaled, w, b)
    for h, p, pr in zip(test_hours.ravel(), preds.ravel(), probs.ravel()):
        print(f"Hours={h:.1f} --> Predicted={p} (prob={pr:.3f})")
