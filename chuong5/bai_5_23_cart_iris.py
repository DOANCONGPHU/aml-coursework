"""Bài 5.23 - Thực hành CART (DecisionTreeClassifier, tiêu chí Gini) trên tập Iris.

Thí nghiệm:
1. Cây không giới hạn độ sâu (max_depth=None) -> vẽ cây, nhận xét các nút.
2. max_depth = 2 và 3 -> so sánh accuracy train/test, nhận xét quá khớp.

Chạy:  python -m chuong5.bai_5_23_cart_iris
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def run_experiment(depths=(None, 2, 3), seed=42, test_size=0.3):
    """Huấn luyện cây CART với từng giá trị max_depth và trả về bảng kết quả.

    Mỗi phần tử kết quả gồm: độ sâu thực tế, số lá, accuracy train, accuracy test
    và chính mô hình (để vẽ). Dùng stratify để giữ tỉ lệ 3 lớp ở train/test.
    """
    iris = load_iris()
    X_tr, X_te, y_tr, y_te = train_test_split(
        iris.data, iris.target, test_size=test_size, random_state=seed, stratify=iris.target)
    rows = []
    for d in depths:
        clf = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=0).fit(X_tr, y_tr)
        rows.append({"max_depth": d, "depth": clf.get_depth(), "leaves": clf.get_n_leaves(),
                     "train_acc": clf.score(X_tr, y_tr), "test_acc": clf.score(X_te, y_te),
                     "model": clf})
    return rows, iris


def main():
    """In bảng so sánh, in luật của cây đầy đủ và lưu hình cây vào figures/."""
    rows, iris = run_experiment()
    print(f"{'max_depth':>9s} {'depth':>5s} {'leaves':>6s} {'train':>6s} {'test':>6s}")
    for r in rows:
        print(f"{str(r['max_depth']):>9s} {r['depth']:5d} {r['leaves']:6d} "
              f"{r['train_acc']:6.3f} {r['test_acc']:6.3f}")
    full = rows[0]["model"]
    print(export_text(full, feature_names=list(iris.feature_names)))

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 10))
    plot_tree(full, feature_names=iris.feature_names, class_names=iris.target_names,
              filled=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title("CART (Gini) trên Iris - không giới hạn độ sâu")
    fig.savefig(os.path.join(FIG_DIR, "5_23_iris_tree.png"), dpi=90, bbox_inches="tight")
    plt.close(fig)
    return rows


if __name__ == "__main__":
    main()
