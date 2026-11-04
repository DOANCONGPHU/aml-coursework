"""Bài 5.24 - Cây hồi quy CART trên dữ liệu giả lập y = 2x + sin(x) + nhiễu.

- Sinh 50 điểm huấn luyện, huấn luyện DecisionTreeRegressor với max_depth = 2, 5, 10.
- Vẽ đường dự đoán của từng mô hình cùng dữ liệu gốc lên một đồ thị.
- Sinh thêm dữ liệu mới (test) để đánh giá khả năng tổng quát hóa (MSE train vs test).

Chạy:  python -m chuong5.bai_5_24_cart_hoi_quy
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def make_data(n=50, noise=1.0, seed=0, low=0.0, high=10.0):
    """Sinh n điểm x ~ U(low, high) và y = 2x + sin(x) + N(0, noise^2).
    Trả về X dạng cột (n, 1) để đưa thẳng vào sklearn và y dạng vector."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(low, high, n))
    y = 2 * x + np.sin(x) + rng.normal(0, noise, n)
    return x.reshape(-1, 1), y


def fit_trees(X, y, depths=(2, 5, 10)):
    """Huấn luyện một cây hồi quy cho mỗi max_depth, trả về dict {depth: model}."""
    return {d: DecisionTreeRegressor(max_depth=d, random_state=0).fit(X, y) for d in depths}


def evaluate(models, X_tr, y_tr, X_te, y_te):
    """Tính MSE trên train và test cho từng mô hình -> dict {depth: (mse_train, mse_test)}."""
    return {d: (mean_squared_error(y_tr, m.predict(X_tr)), mean_squared_error(y_te, m.predict(X_te)))
            for d, m in models.items()}


def main():
    """Chạy thí nghiệm, in MSE và lưu đồ thị so sánh 3 độ sâu vào figures/."""
    X_tr, y_tr = make_data(50, seed=0)
    X_te, y_te = make_data(500, seed=1)          # dữ liệu "mới" cùng phân phối
    models = fit_trees(X_tr, y_tr)
    scores = evaluate(models, X_tr, y_tr, X_te, y_te)
    print(f"{'max_depth':>9s} {'MSE train':>10s} {'MSE test':>10s} {'số lá':>6s}")
    for d, (a, b) in scores.items():
        print(f"{d:9d} {a:10.3f} {b:10.3f} {models[d].get_n_leaves():6d}")

    os.makedirs(FIG_DIR, exist_ok=True)
    grid = np.linspace(0, 10, 1000).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(X_tr, y_tr, c="k", s=20, label="dữ liệu huấn luyện (50 điểm)")
    ax.plot(grid, 2 * grid.ravel() + np.sin(grid.ravel()), "k:", label="hàm thật 2x + sin x")
    for d, m in models.items():
        ax.plot(grid, m.predict(grid), label=f"max_depth={d}")
    ax.set(xlabel="x", ylabel="y", title="Cây hồi quy CART với các độ sâu khác nhau")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "5_24_cart_hoi_quy.png"), dpi=110)
    plt.close(fig)
    return scores


if __name__ == "__main__":
    main()
