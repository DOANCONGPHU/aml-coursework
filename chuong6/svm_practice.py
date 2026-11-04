"""Chương 6 - Bài 4: Thực hành SVM trên tập Breast Cancer (sklearn).

a) SVM tuyến tính với C = 0.1, 1, 10: so sánh số support vectors và accuracy test.
b) SVM kernel RBF với các cặp (C, gamma): vẽ đường biên trên 2 thành phần chính (PCA)
   và nhận xét hiện tượng overfitting.

Ngoài ra có hàm `hard_margin_svm` dùng để kiểm chứng Bài 3 & Bài 5 (tính tay).

Chạy:  python -m chuong6.svm_practice
"""
import os

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def hard_margin_svm(X, y):
    """Xấp xỉ SVM lề cứng bằng SVC tuyến tính với C rất lớn (phạt vi phạm ~ vô cùng).

    Trả về dict: w, b, margin = 2/||w||, chỉ số support vectors và hệ số alpha_i
    (dual_coef_ = alpha_i * y_i). Dùng để đối chiếu với lời giải hình học.
    """
    clf = SVC(kernel="linear", C=1e6).fit(np.asarray(X, float), y)
    w = clf.coef_[0]
    return {"w": w, "b": float(clf.intercept_[0]), "margin": 2 / np.linalg.norm(w),
            "support": sorted(clf.support_.tolist()),
            "alpha": dict(zip(clf.support_.tolist(), np.abs(clf.dual_coef_[0]).tolist()))}


def load_data(seed=42):
    """Tải Breast Cancer, chia train/test 75/25 có stratify."""
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)


def linear_svm_study(Cs=(0.1, 1, 10)):
    """Câu a: với mỗi C, huấn luyện SVM tuyến tính (đã chuẩn hóa dữ liệu) và ghi lại
    số support vectors, accuracy train/test. Trả về list dict kết quả."""
    X_tr, X_te, y_tr, y_te = load_data()
    rows = []
    for C in Cs:
        model = make_pipeline(StandardScaler(), SVC(kernel="linear", C=C)).fit(X_tr, y_tr)
        svc = model[-1]
        rows.append({"C": C, "n_sv": int(svc.n_support_.sum()),
                     "train": model.score(X_tr, y_tr), "test": model.score(X_te, y_te)})
    return rows


def rbf_svm_study(params=((1, 0.01), (1, 0.1), (1, 1), (100, 1), (1000, 10))):
    """Câu b: SVM RBF trên toàn bộ 30 đặc trưng với các cặp (C, gamma).
    Trả về list dict (C, gamma, số SV, acc train, acc test)."""
    X_tr, X_te, y_tr, y_te = load_data()
    rows = []
    for C, g in params:
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma=g)).fit(X_tr, y_tr)
        rows.append({"C": C, "gamma": g, "n_sv": int(model[-1].n_support_.sum()),
                     "train": model.score(X_tr, y_tr), "test": model.score(X_te, y_te)})
    return rows


def plot_rbf_boundaries(params=((1, 0.1), (10, 1), (1000, 10))):
    """Câu b (vẽ): chuẩn hóa -> PCA 2 chiều -> huấn luyện SVM RBF trên 2 chiều đó và
    tô màu vùng quyết định. Gamma/C lớn sinh đường biên ngoằn ngoèo bám sát từng điểm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X_tr, X_te, y_tr, y_te = load_data()
    prep = make_pipeline(StandardScaler(), PCA(n_components=2)).fit(X_tr)
    P_tr, P_te = prep.transform(X_tr), prep.transform(X_te)
    xx, yy = np.meshgrid(np.linspace(P_tr[:, 0].min() - 1, P_tr[:, 0].max() + 1, 300),
                         np.linspace(P_tr[:, 1].min() - 1, P_tr[:, 1].max() + 1, 300))
    fig, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 4.5))
    scores = []
    for ax, (C, g) in zip(axes, params):
        clf = SVC(kernel="rbf", C=C, gamma=g).fit(P_tr, y_tr)
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z > 0, alpha=0.25, cmap="coolwarm")
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"])
        ax.scatter(P_tr[:, 0], P_tr[:, 1], c=y_tr, cmap="coolwarm", s=10, edgecolors="k", linewidths=0.2)
        tr, te = clf.score(P_tr, y_tr), clf.score(P_te, y_te)
        scores.append({"C": C, "gamma": g, "train": tr, "test": te, "n_sv": int(clf.n_support_.sum())})
        ax.set_title(f"C={C}, γ={g}\ntrain={tr:.3f}  test={te:.3f}  SV={scores[-1]['n_sv']}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "6_bai4_svm_rbf_pca.png"), dpi=100)
    plt.close(fig)
    return scores


def main():
    """In kết quả câu a, câu b và lưu hình đường biên PCA."""
    print("a) SVM tuyến tính")
    for r in linear_svm_study():
        print(f"   C={r['C']:<5} SV={r['n_sv']:4d}  train={r['train']:.3f}  test={r['test']:.3f}")
    print("b) SVM RBF (30 đặc trưng)")
    for r in rbf_svm_study():
        print(f"   C={r['C']:<5} gamma={r['gamma']:<5} SV={r['n_sv']:4d}  "
              f"train={r['train']:.3f}  test={r['test']:.3f}")
    print("b) SVM RBF trên 2 thành phần PCA")
    for r in plot_rbf_boundaries():
        print(f"   C={r['C']:<5} gamma={r['gamma']:<5} SV={r['n_sv']:4d}  "
              f"train={r['train']:.3f}  test={r['test']:.3f}")


if __name__ == "__main__":
    main()
