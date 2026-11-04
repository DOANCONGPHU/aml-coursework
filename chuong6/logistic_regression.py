"""Chương 6 - Bài 1 & Bài 2: Hồi quy logistic cài đặt bằng gradient descent (không dùng sklearn).

Mô hình:  ŷ = σ(w·x + b),  σ(z) = 1 / (1 + e^{-z})
Hàm mất mát (Binary Cross-Entropy trung bình):
    J(w, b) = -1/N * sum [ y log ŷ + (1 - y) log(1 - ŷ) ]
Gradient:
    dJ/dw = 1/N * X^T (ŷ - y),   dJ/db = 1/N * sum (ŷ - y)

Chạy Bài 1:  python -m chuong6.logistic_regression bai1
Chạy Bài 2:  python -m chuong6.logistic_regression bai2
"""
import os
import sys

import numpy as np

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def sigmoid(z):
    """Hàm sigmoid đưa số thực bất kỳ về khoảng (0, 1) để biểu diễn xác suất.
    Cắt z vào [-500, 500] để tránh tràn số khi tính exp."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def predict_proba(X, w, b):
    """Xác suất thuộc lớp 1: ŷ = σ(Xw + b). X shape (N, d), w shape (d,)."""
    return sigmoid(np.asarray(X, float) @ np.asarray(w, float) + b)


def bce_loss(y, y_hat, eps=1e-12):
    """Binary Cross-Entropy trung bình. Kẹp ŷ trong [eps, 1-eps] để tránh log(0)."""
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return float(-np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))


def gradients(X, y, w, b):
    """Tính gradient của BCE theo w và b trên toàn bộ dữ liệu.

    Sai số mỗi mẫu e_i = ŷ_i - y_i; dw = X^T e / N; db = mean(e).
    Khi ŷ_i ≈ y_i (dự đoán đúng và tự tin) thì e_i ≈ 0 -> mẫu gần như không đóng góp.
    """
    X = np.asarray(X, float)
    err = predict_proba(X, w, b) - y
    return X.T @ err / len(y), float(np.mean(err))


def gd_step(X, y, w, b, lr):
    """Một bước (một epoch full-batch) gradient descent: w -= lr*dw, b -= lr*db."""
    dw, db = gradients(X, y, w, b)
    return w - lr * dw, b - lr * db


def fit(X, y, lr=0.1, epochs=1000, w0=None, b0=0.0):
    """Huấn luyện hồi quy logistic bằng full-batch gradient descent.

    Trả về (w, b, lịch sử loss theo epoch). Lịch sử dùng để vẽ đồ thị loss-epoch.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    w = np.zeros(X.shape[1]) if w0 is None else np.asarray(w0, float).copy()
    b = float(b0)
    history = [bce_loss(y, predict_proba(X, w, b))]
    for _ in range(epochs):
        w, b = gd_step(X, y, w, b, lr)
        history.append(bce_loss(y, predict_proba(X, w, b)))
    return w, b, history


def predict(X, w, b, threshold=0.5):
    """Gán nhãn 1 nếu xác suất ≥ ngưỡng (mặc định 0.5), ngược lại 0."""
    return (predict_proba(X, w, b) >= threshold).astype(int)


# ------------------------------------------------------------------------------ Bài 1
def bai_1():
    """Bài 1: 3 điểm (1,0), (2,0), (4,1); w = 0.5, b = -1.5; lr = 0.5.
    In ŷ từng điểm, loss trung bình, gradient và (w, b) sau một epoch."""
    X = np.array([[1.0], [2.0], [4.0]])
    y = np.array([0.0, 0.0, 1.0])
    w, b = np.array([0.5]), -1.5
    y_hat = predict_proba(X, w, b)
    dw, db = gradients(X, y, w, b)
    w1, b1 = gd_step(X, y, w, b, lr=0.5)
    print("a) ŷ =", np.round(y_hat, 4))
    print("b) J  =", round(bce_loss(y, y_hat), 4))
    print("c) dJ/dw =", round(float(dw[0]), 4), " dJ/db =", round(db, 4))
    print("d) w_new =", round(float(w1[0]), 4), " b_new =", round(b1, 4))
    return y_hat, bce_loss(y, y_hat), (float(dw[0]), db), (float(w1[0]), b1)


# ------------------------------------------------------------------------------ Bài 2
def load_iris_2class():
    """Lấy 2 thuộc tính đầu (sepal length, sepal width) của 2 lớp setosa (0)
    và versicolor (1) trong tập Iris -> 100 mẫu."""
    from sklearn.datasets import load_iris   # chỉ dùng sklearn để TẢI dữ liệu
    iris = load_iris()
    mask = iris.target < 2
    return iris.data[mask, :2], iris.target[mask].astype(float)


def bai_2(lrs=(0.001, 0.01, 0.1, 1.0), epochs=3000):
    """Bài 2: huấn luyện với nhiều learning rate, vẽ loss-epoch và đường biên.

    Chuẩn hóa đặc trưng (z-score) giúp gradient descent hội tụ nhanh và ổn định.
    Đường biên quyết định là tập điểm có w·x + b = 0 (tương ứng ŷ = 0.5).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, y = load_iris_2class()
    mu, sd = X.mean(0), X.std(0)
    Xs = (X - mu) / sd
    results = {}
    for lr in lrs:
        w, b, hist = fit(Xs, y, lr=lr, epochs=epochs)
        acc = float(np.mean(predict(Xs, w, b) == y))
        results[lr] = {"w": w, "b": b, "history": hist, "acc": acc}
        print(f"lr={lr:<6} loss cuối={hist[-1]:.4f}  accuracy={acc:.2%}  "
              f"loss@100={hist[100]:.4f}")

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for lr, r in results.items():
        ax[0].plot(r["history"], label=f"lr = {lr}")
    ax[0].set(xscale="log", xlabel="epoch (log)", ylabel="BCE loss", title="Loss theo epoch")
    ax[0].legend()

    best = results[max(lrs)]
    for cls, mk, name in [(0, "o", "setosa"), (1, "s", "versicolor")]:
        ax[1].scatter(X[y == cls, 0], X[y == cls, 1], marker=mk, label=name, alpha=0.8)
    xs = np.linspace(X[:, 0].min() - 0.3, X[:, 0].max() + 0.3, 100)
    # w1*(x1-mu1)/sd1 + w2*(x2-mu2)/sd2 + b = 0  =>  giải x2 theo x1 (trên thang gốc)
    w1, w2 = best["w"]
    x2 = mu[1] - sd[1] / w2 * (w1 * (xs - mu[0]) / sd[0] + best["b"])
    ax[1].plot(xs, x2, "k--", label=f"đường biên (lr={max(lrs)})")
    ax[1].set(xlabel="sepal length (cm)", ylabel="sepal width (cm)",
              ylim=(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5), title="Đường biên phân lớp")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "6_bai2_logistic.png"), dpi=110)
    plt.close(fig)
    return results


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("bai1", "all"):
        bai_1()
    if which in ("bai2", "all"):
        bai_2()
