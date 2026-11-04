"""Bài 5.25 - Cây phân lớp & cây hồi quy trên dữ liệu khách hàng đọc từ file CSV (tự tạo).

Các bước:
1. `generate_customers_csv`: sinh dữ liệu khách hàng giả lập và ghi ra CSV.
2. Đọc CSV bằng pandas, mã hóa one-hot các cột hạng mục.
3. Cây phân lớp: dự đoán khách có mua sản phẩm (`mua_hang`) hay không.
   Cây hồi quy: dự đoán số tiền chi tiêu (`chi_tieu`, biến mục tiêu số).
4. Cắt tỉa trước (pre-pruning) bằng cách thay đổi `min_samples_leaf`.
5. In tầm quan trọng đặc trưng (feature importance) và lưu hình cây thành file ảnh.

Chạy:  python -m chuong5.bai_5_25_khach_hang
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

BASE = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE, "data", "khach_hang.csv")
FIG_DIR = os.path.join(BASE, "..", "figures")


def generate_customers_csv(path=CSV_PATH, n=600, seed=7):
    """Sinh n khách hàng giả lập với quy luật ẩn rồi ghi ra file CSV.

    Cột: tuoi, thu_nhap (triệu/tháng), so_lan_mua_truoc, thanh_vien (Có/Không),
    khu_vuc (Bắc/Trung/Nam), chi_tieu (nghìn đồng, số), mua_hang (0/1, nhãn lớp).
    Chi tiêu phụ thuộc thu nhập, số lần mua trước, thẻ thành viên + nhiễu; khách mua
    hàng khi điểm xu hướng (phụ thuộc thu nhập, tuổi, thành viên) vượt ngưỡng.
    """
    rng = np.random.default_rng(seed)
    tuoi = rng.integers(18, 70, n)
    thu_nhap = np.round(rng.gamma(4.0, 5.0, n), 1)
    so_lan = rng.poisson(3, n)
    thanh_vien = rng.choice(["Có", "Không"], n, p=[0.4, 0.6])
    khu_vuc = rng.choice(["Bắc", "Trung", "Nam"], n)
    tv = (thanh_vien == "Có").astype(float)
    chi_tieu = np.round(50 + 12 * thu_nhap + 30 * so_lan + 150 * tv + rng.normal(0, 60, n), 0)
    diem = 0.08 * thu_nhap + 1.2 * tv - 0.03 * np.abs(tuoi - 35) + 0.2 * so_lan + rng.normal(0, 0.6, n)
    mua_hang = (diem > 1.5).astype(int)
    df = pd.DataFrame({"tuoi": tuoi, "thu_nhap": thu_nhap, "so_lan_mua_truoc": so_lan,
                       "thanh_vien": thanh_vien, "khu_vuc": khu_vuc,
                       "chi_tieu": chi_tieu, "mua_hang": mua_hang})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return df


def load_features(path=CSV_PATH):
    """Đọc CSV, tách đặc trưng X (one-hot cho cột chữ) và hai biến mục tiêu.
    Trả về (X, y_lop, y_so)."""
    df = pd.read_csv(path)
    X = pd.get_dummies(df.drop(columns=["chi_tieu", "mua_hang"]),
                       columns=["thanh_vien", "khu_vuc"], drop_first=True, dtype=int)
    return X, df["mua_hang"], df["chi_tieu"]


def pruning_study(model_cls, X, y, leaves=(1, 5, 20, 50), seed=0):
    """Huấn luyện cùng một loại cây với nhiều giá trị `min_samples_leaf`.

    min_samples_leaf lớn -> mỗi lá phải chứa nhiều mẫu -> cây nông, ít lá hơn
    (cắt tỉa trước). Trả về list (min_samples_leaf, số lá, điểm train, điểm CV, điểm test,
    model) với điểm là accuracy (phân lớp) hoặc R2 (hồi quy). Điểm CV là trung bình
    cross-validation 5-fold trên tập train - dùng để CHỌN tham số (không nhìn tập test).
    """
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed)
    is_cls = model_cls is DecisionTreeClassifier
    metric = accuracy_score if is_cls else r2_score
    out = []
    for m in leaves:
        mdl = model_cls(min_samples_leaf=m, random_state=0)
        cv = cross_val_score(mdl, X_tr, y_tr, cv=5, scoring="accuracy" if is_cls else "r2").mean()
        mdl.fit(X_tr, y_tr)
        out.append((m, mdl.get_n_leaves(), metric(y_tr, mdl.predict(X_tr)), cv,
                    metric(y_te, mdl.predict(X_te)), mdl))
    return out


def save_tree_image(model, feature_names, path, class_names=None, max_depth=3):
    """Vẽ cây (giới hạn số tầng hiển thị cho dễ đọc) và lưu thành file PNG."""
    fig, ax = plt.subplots(figsize=(20, 9))
    plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True,
              rounded=True, max_depth=max_depth, fontsize=8, ax=ax)
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def main():
    """Chạy toàn bộ bài 5.25: sinh CSV, cắt tỉa, in feature importance, lưu hình cây."""
    generate_customers_csv()
    X, y_cls, y_reg = load_features()
    os.makedirs(FIG_DIR, exist_ok=True)
    chosen = {}
    for title, cls, y, metric_name in [("CÂY PHÂN LỚP (mua_hang)", DecisionTreeClassifier, y_cls, "acc"),
                                       ("CÂY HỒI QUY (chi_tieu)", DecisionTreeRegressor, y_reg, "R2")]:
        print(f"\n=== {title} ===")
        print(f"{'min_samples_leaf':>16s} {'số lá':>6s} {metric_name + ' train':>10s} "
              f"{metric_name + ' CV':>10s} {metric_name + ' test':>10s}")
        res = pruning_study(cls, X, y)
        for m, nl, tr, cv, te, _ in res:
            print(f"{m:16d} {nl:6d} {tr:10.3f} {cv:10.3f} {te:10.3f}")
        best = max(res, key=lambda r: r[3])
        model = best[5]
        chosen[title] = best
        print(f"-> chọn min_samples_leaf = {best[0]} (điểm CV 5-fold cao nhất)")
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("Tầm quan trọng đặc trưng:\n" + imp.round(3).to_string())
        name = "5_25_cay_phan_lop.png" if cls is DecisionTreeClassifier else "5_25_cay_hoi_quy.png"
        save_tree_image(model, list(X.columns), os.path.join(FIG_DIR, name),
                        class_names=["Không mua", "Mua"] if cls is DecisionTreeClassifier else None)
    return chosen


if __name__ == "__main__":
    main()
