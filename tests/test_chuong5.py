"""Kiểm thử Chương 5: entropy, Gini, Information Gain, cây ID3 và các bài thực hành CART."""
import os

import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from chuong5.id3 import ATTRIBUTES, DATA, TARGET, build_tree, entropy, information_gain, predict


def gini(labels):
    """Chỉ số Gini = 1 - sum p_c^2 (dùng để kiểm chứng câu 5.3, 5.16)."""
    n = len(labels)
    return 1 - sum((labels.count(c) / n) ** 2 for c in set(labels))


# ---------------------------------------------------------------- trắc nghiệm có tính toán
def test_cau_5_1_entropy_70_30():
    assert entropy(["a"] * 7 + ["b"] * 3) == pytest.approx(0.881, abs=1e-3)


def test_cau_5_11_entropy_mot_lop_bang_0():
    assert entropy(["x"] * 9) == 0


def test_cau_5_14_entropy_6_4():
    assert entropy(["Có"] * 6 + ["Không"] * 4) == pytest.approx(0.971, abs=1e-3)


def test_cau_5_3_va_5_16_gini():
    assert gini(["A"] * 5) == 0
    assert gini(["A", "A", "A", "B"]) == pytest.approx(0.375)


# ---------------------------------------------------------------- bài 5.21 (du lịch)
DU_LICH = [dict(zip(["Thời tiết", "Gió", "Đi"], r)) for r in [
    ("Nắng", "Mạnh", "Không"), ("Nắng", "Nhẹ", "Có"), ("Mưa", "Mạnh", "Không"),
    ("Mưa", "Nhẹ", "Có"), ("Âm u", "Mạnh", "Có")]]


def test_bai_5_21_entropy_va_gain():
    assert entropy([r["Đi"] for r in DU_LICH]) == pytest.approx(0.971, abs=1e-3)
    assert information_gain(DU_LICH, "Thời tiết", "Đi") == pytest.approx(0.171, abs=1e-3)
    assert information_gain(DU_LICH, "Gió", "Đi") == pytest.approx(0.420, abs=1e-3)


def test_bai_5_21_cay_id3():
    tree = build_tree(DU_LICH, ["Thời tiết", "Gió"], "Đi")
    assert "Gió" in tree                                      # gốc là Gió (Gain lớn nhất)
    assert tree["Gió"]["Nhẹ"] == "Có"
    assert tree["Gió"]["Mạnh"]["Thời tiết"] == {"Nắng": "Không", "Mưa": "Không", "Âm u": "Có"}


# ---------------------------------------------------------------- bài 5.22 (ID3 mua máy tính)
def test_bai_5_22_gain_tung_thuoc_tinh():
    assert entropy([r[TARGET] for r in DATA]) == pytest.approx(0.940, abs=1e-3)
    gains = {a: information_gain(DATA, a) for a in ATTRIBUTES}
    assert gains["Tuổi"] == pytest.approx(0.247, abs=1e-3)
    assert gains["Sinh viên"] == pytest.approx(0.152, abs=1e-3)
    assert max(gains, key=gains.get) == "Tuổi"


def test_bai_5_22_cau_truc_cay():
    tree = build_tree(DATA, ATTRIBUTES)
    root = tree["Tuổi"]
    assert root["Trung niên"] == "Có"
    assert root["Trẻ"]["Sinh viên"] == {"Không": "Không", "Có": "Có"}
    assert root["Già"]["Đánh giá tín dụng"] == {"Tốt": "Có", "Xuất sắc": "Không"}


def test_bai_5_22_predict():
    tree = build_tree(DATA, ATTRIBUTES)
    assert all(predict(tree, r) == r[TARGET] for r in DATA)    # khớp 100% tập huấn luyện
    assert predict(tree, {"Tuổi": "Trẻ", "Sinh viên": "Có"}) == "Có"
    assert predict(tree, {"Tuổi": "Già", "Đánh giá tín dụng": "Xuất sắc"}) == "Không"
    # Giá trị lạ -> trả về nhãn đa số của nút gốc (9 Có / 5 Không)
    assert predict(tree, {"Tuổi": "Thiếu niên"}) == "Có"


# ---------------------------------------------------------------- bài 5.23 - 5.25
def test_bai_5_23_cay_sau_hon_thi_train_acc_khong_giam():
    from chuong5.bai_5_23_cart_iris import run_experiment
    rows, _ = run_experiment()
    by = {r["max_depth"]: r for r in rows}
    assert by[None]["train_acc"] == 1.0
    assert by[2]["train_acc"] <= by[3]["train_acc"] <= by[None]["train_acc"]
    assert by[2]["depth"] == 2 and by[3]["depth"] == 3


def test_bai_5_24_du_lieu_va_qua_khop():
    from chuong5.bai_5_24_cart_hoi_quy import evaluate, fit_trees, make_data
    X, y = make_data(50)
    assert X.shape == (50, 1) and y.shape == (50,)
    X_te, y_te = make_data(500, seed=1)
    s = evaluate(fit_trees(X, y), X, y, X_te, y_te)
    assert s[10][0] < s[5][0] < s[2][0]           # sâu hơn -> MSE train nhỏ hơn
    assert s[10][1] > s[10][0]                     # độ sâu 10: test tệ hơn train nhiều (quá khớp)


def test_bai_5_25_csv_va_cat_tia(tmp_path):
    from chuong5.bai_5_25_khach_hang import generate_customers_csv, load_features, pruning_study
    path = tmp_path / "kh.csv"
    df = generate_customers_csv(str(path), n=200)
    assert os.path.exists(path) and len(df) == 200
    X, y_cls, y_reg = load_features(str(path))
    assert "thanh_vien_Không" in X.columns and "chi_tieu" not in X.columns
    res = pruning_study(DecisionTreeClassifier, X, y_cls, leaves=(1, 20))
    assert res[1][1] < res[0][1]                    # min_samples_leaf lớn -> ít lá hơn
    res_r = pruning_study(DecisionTreeRegressor, X, y_reg, leaves=(1, 20))
    assert res_r[0][2] == pytest.approx(1.0)        # cây không tỉa khớp hoàn toàn tập train
    assert res_r[0][3] < res_r[0][2]                # điểm CV < điểm train (dấu hiệu quá khớp)
