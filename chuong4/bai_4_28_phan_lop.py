"""Bài 4.28 - Bài toán phân lớp bằng mạng nơ-ron trên tập Breast Cancer Wisconsin (sklearn).

Tập dữ liệu: 569 khối u, 30 đặc trưng số; nhãn 0 = ác tính (malignant), 1 = lành tính (benign).

Quy trình:
1. Chia train/test 80/20 có phân tầng (stratify) để giữ tỉ lệ lớp.
2. MLP tự cài đặt (chuong4.mlp_numpy).
3. MLPClassifier + GridSearchCV (5-fold, tối ưu F1).
4. Đánh giá Accuracy, Precision, Recall, F1 và vẽ ma trận nhầm lẫn.

Chạy:  python -m chuong4.bai_4_28_phan_lop
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chuong4.metrics import classification_report_binary
from chuong4.mlp_numpy import MLP

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def load_data(seed=42):
    """Tải Breast Cancer và chia train/test 80/20 có stratify theo nhãn."""
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def train_own_mlp(X_tr, y_tr):
    """Huấn luyện MLP tự cài đặt (2 tầng ẩn 16-8, ReLU, sigmoid ở đầu ra, BCE loss)."""
    sx = StandardScaler().fit(X_tr)
    net = MLP(hidden_layers=(16, 8), task="classification", lr=0.05, epochs=200,
              batch_size=32, l2=0.5, random_state=0).fit(sx.transform(X_tr), y_tr)
    return (lambda X: net.predict(sx.transform(X))), net


def tune_sklearn_mlp(X_tr, y_tr):
    """GridSearchCV 5-fold cho MLPClassifier, tiêu chí chọn mô hình là F1."""
    pipe = make_pipeline(StandardScaler(),
                         MLPClassifier(max_iter=2000, random_state=0))
    grid = {
        "mlpclassifier__hidden_layer_sizes": [(8,), (16,), (32, 16)],
        "mlpclassifier__alpha": [1e-4, 1e-2, 1.0],
        "mlpclassifier__activation": ["relu", "tanh"],
    }
    gs = GridSearchCV(pipe, grid, cv=5, scoring="f1", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    return gs


def main():
    """Chạy thí nghiệm, in bảng độ đo trên tập test và lưu ma trận nhầm lẫn."""
    X_tr, X_te, y_tr, y_te = load_data()
    own_predict, _ = train_own_mlp(X_tr, y_tr)
    gs = tune_sklearn_mlp(X_tr, y_tr)
    results = {
        "MLP tự cài đặt (16,8)": classification_report_binary(y_te, own_predict(X_te)),
        "MLPClassifier tốt nhất": classification_report_binary(y_te, gs.predict(X_te)),
    }
    print("Tham số tốt nhất:", gs.best_params_)
    print(f"{'Mô hình':26s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
    for name, r in results.items():
        print(f"{name:26s} {r['Accuracy']:6.3f} {r['Precision']:6.3f} "
              f"{r['Recall']:6.3f} {r['F1']:6.3f}")

    os.makedirs(FIG_DIR, exist_ok=True)
    disp = ConfusionMatrixDisplay.from_estimator(
        gs.best_estimator_, X_te, y_te, display_labels=["ác tính", "lành tính"], cmap="Blues")
    disp.ax_.set_title("Ma trận nhầm lẫn - MLPClassifier (test)")
    disp.figure_.tight_layout()
    disp.figure_.savefig(os.path.join(FIG_DIR, "4_28_confusion.png"), dpi=110)
    plt.close("all")
    return results, gs.best_params_


if __name__ == "__main__":
    main()
