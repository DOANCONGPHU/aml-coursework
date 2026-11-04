"""Bài 4.27 - Bài toán hồi quy bằng mạng nơ-ron trên tập Diabetes (sklearn).

Tập Diabetes: 442 bệnh nhân, 10 đặc trưng (tuổi, giới tính, BMI, huyết áp, 6 chỉ số máu),
mục tiêu là mức độ tiến triển của bệnh sau 1 năm (số thực).

Quy trình:
1. Chia train/test 80/20, chuẩn hóa đặc trưng (StandardScaler) - rất quan trọng với NN.
2. Baseline: Linear Regression.
3. MLP tự cài đặt (chuong4.mlp_numpy).
4. MLPRegressor của sklearn + GridSearchCV (5-fold) để chọn kiến trúc & điều chuẩn tốt nhất.
5. Đánh giá bằng R2, NSE, MAE, RMSE trên tập test, vẽ biểu đồ dự đoán.

Chạy:  python -m chuong4.bai_4_27_hoi_quy
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chuong4.metrics import regression_report
from chuong4.mlp_numpy import MLP

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def load_data(seed=42):
    """Tải tập Diabetes và chia train/test 80/20 với hạt giống cố định."""
    X, y = load_diabetes(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=seed)


def train_own_mlp(X_tr, y_tr):
    """Huấn luyện MLP tự cài đặt. Chuẩn hóa cả y (trung bình 0, độ lệch 1) để
    gradient ổn định, khi dự đoán thì biến đổi ngược lại về thang gốc."""
    sx = StandardScaler().fit(X_tr)
    y_mu, y_sd = y_tr.mean(), y_tr.std()
    net = MLP(hidden_layers=(16, 8), task="regression", activation="relu", lr=0.01,
              epochs=300, batch_size=32, l2=1.0, random_state=0)
    net.fit(sx.transform(X_tr), (y_tr - y_mu) / y_sd)
    return lambda X: net.predict(sx.transform(X)) * y_sd + y_mu, net


def tune_sklearn_mlp(X_tr, y_tr):
    """Tìm siêu tham số tốt nhất cho MLPRegressor bằng GridSearchCV 5-fold.

    Lưới tìm kiếm: kiến trúc tầng ẩn, hệ số L2 (alpha), learning rate khởi tạo.
    Dùng early_stopping để dừng khi điểm validation không còn cải thiện.
    """
    pipe = make_pipeline(StandardScaler(),
                         MLPRegressor(max_iter=3000, early_stopping=True,
                                      n_iter_no_change=30, random_state=0))
    grid = {
        "mlpregressor__hidden_layer_sizes": [(8,), (16,), (32, 16), (64, 32)],
        "mlpregressor__alpha": [1e-3, 1e-1, 1.0, 10.0],
        "mlpregressor__learning_rate_init": [1e-3, 1e-2],
    }
    gs = GridSearchCV(pipe, grid, cv=5, scoring="r2", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    return gs


def main():
    """Chạy toàn bộ thí nghiệm, in bảng kết quả và lưu hình vào thư mục figures/."""
    X_tr, X_te, y_tr, y_te = load_data()
    results = {}

    lin = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
    results["Linear Regression (baseline)"] = regression_report(y_te, lin.predict(X_te))

    own_predict, own_net = train_own_mlp(X_tr, y_tr)
    results["MLP tự cài đặt (16,8)"] = regression_report(y_te, own_predict(X_te))

    gs = tune_sklearn_mlp(X_tr, y_tr)
    best_pred = gs.predict(X_te)
    results["MLPRegressor tốt nhất (GridSearch)"] = regression_report(y_te, best_pred)

    print("Tham số tốt nhất:", gs.best_params_)
    print(f"{'Mô hình':38s} {'R2':>7s} {'NSE':>7s} {'MAE':>7s} {'RMSE':>7s}")
    for name, r in results.items():
        print(f"{name:38s} {r['R2']:7.3f} {r['NSE']:7.3f} {r['MAE']:7.2f} {r['RMSE']:7.2f}")

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(own_net.loss_history_)
    ax[0].set(title="Loss của MLP tự cài đặt", xlabel="epoch", ylabel="MSE/2 (y chuẩn hóa)")
    ax[1].scatter(y_te, best_pred, alpha=0.7)
    lo, hi = y_te.min(), y_te.max()
    ax[1].plot([lo, hi], [lo, hi], "r--", label="y = ŷ")
    ax[1].set(title="Diabetes - thực tế vs dự đoán (test)", xlabel="y thực", ylabel="y dự đoán")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "4_27_hoi_quy.png"), dpi=110)
    return results, gs.best_params_


if __name__ == "__main__":
    main()
