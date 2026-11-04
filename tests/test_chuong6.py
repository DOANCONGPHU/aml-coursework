"""Kiểm thử Chương 6: hồi quy logistic (Bài 1, 2) và SVM (Bài 3, 4, 5)."""
import numpy as np
import pytest

from chuong6.logistic_regression import (bai_1, bce_loss, fit, gradients, load_iris_2class,
                                         predict, sigmoid)
from chuong6.svm_practice import hard_margin_svm, linear_svm_study, rbf_svm_study


# ------------------------------------------------------------------ trắc nghiệm có tính toán
def test_cau_6_1_sigmoid_trong_khoang_0_1():
    z = np.linspace(-30, 30, 101)
    s = sigmoid(z)
    assert np.all((s > 0) & (s < 1))
    assert sigmoid(0) == 0.5


def test_cau_6_4_va_6_13_gradient_gan_0_khi_du_doan_dung():
    X = np.array([[1.0]])
    dw, db = gradients(X, np.array([1.0]), np.array([20.0]), 0.0)   # ŷ ≈ 1, y = 1
    assert abs(dw[0]) < 1e-6 and abs(db) < 1e-6


def test_cau_6_14_cot_toan_0_thi_gradient_0():
    X = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, -1.0]])
    dw, _ = gradients(X, np.array([1.0, 0.0, 1.0]), np.array([0.3, -0.2]), 0.1)
    assert dw[0] == 0.0


# ------------------------------------------------------------------ Bài 1
def test_bai_1_tinh_toan():
    y_hat, loss, (dw, db), (w1, b1) = bai_1()
    np.testing.assert_allclose(y_hat, [0.2689, 0.3775, 0.6225], atol=1e-4)
    assert loss == pytest.approx(0.4205, abs=1e-4)
    assert dw == pytest.approx(-0.1620, abs=1e-4)
    assert db == pytest.approx(0.0896, abs=1e-4)
    assert w1 == pytest.approx(0.5810, abs=1e-4)
    assert b1 == pytest.approx(-1.5448, abs=1e-4)


def test_bce_khong_bi_log_0():
    assert np.isfinite(bce_loss(np.array([1.0, 0.0]), np.array([0.0, 1.0])))


# ------------------------------------------------------------------ Bài 2
def test_bai_2_hoi_tu_va_phan_lop_tot():
    X, y = load_iris_2class()
    assert X.shape == (100, 2)
    Xs = (X - X.mean(0)) / X.std(0)
    w, b, hist = fit(Xs, y, lr=0.1, epochs=2000)
    assert all(a >= b_ - 1e-12 for a, b_ in zip(hist, hist[1:]))   # loss giảm đơn điệu
    assert np.mean(predict(Xs, w, b) == y) >= 0.99


def test_bai_2_lr_nho_hoi_tu_cham_hon():
    X, y = load_iris_2class()
    Xs = (X - X.mean(0)) / X.std(0)
    _, _, h_small = fit(Xs, y, lr=0.001, epochs=200)
    _, _, h_big = fit(Xs, y, lr=0.1, epochs=200)
    assert h_big[-1] < h_small[-1]


# ------------------------------------------------------------------ Bài 3 & 5 (SVM tính tay)
def test_bai_3_svm():
    X = [[0, 0], [1, 0], [2, 1], [3, 1]]
    r = hard_margin_svm(X, [1, 1, -1, -1])
    np.testing.assert_allclose(r["w"], [-1, -1], atol=1e-3)
    assert r["b"] == pytest.approx(2, abs=1e-3)
    assert r["margin"] == pytest.approx(np.sqrt(2), abs=1e-3)
    assert r["support"] == [1, 2]                     # (1,0) và (2,1)


def test_bai_5_svm_B_khong_phai_support_vector():
    X = [[1, 1], [2, 2], [2, 0]]                      # A, B, C
    r = hard_margin_svm(X, [1, 1, -1])
    np.testing.assert_allclose(r["w"], [-1, 1], atol=1e-3)
    assert r["b"] == pytest.approx(1, abs=1e-3)
    assert r["margin"] == pytest.approx(np.sqrt(2), abs=1e-3)
    # B nằm trên lề (y(w·x+b) = 1) nhưng alpha_B = 0
    w, b = np.array([-1, 1]), 1
    assert 1 * (w @ np.array([2, 2]) + b) == 1
    assert r["alpha"].get(1, 0.0) == pytest.approx(0.0, abs=1e-3)
    assert r["alpha"][0] == pytest.approx(1.0, abs=1e-3)
    assert r["alpha"][2] == pytest.approx(1.0, abs=1e-3)


# ------------------------------------------------------------------ Bài 4 (SVM thực hành)
def test_bai_4a_C_lon_thi_it_support_vector_hon():
    rows = linear_svm_study()
    n_sv = [r["n_sv"] for r in rows]                  # C = 0.1, 1, 10
    assert n_sv[0] > n_sv[1] > n_sv[2]
    assert all(r["test"] > 0.9 for r in rows)


def test_bai_4b_gamma_lon_gay_qua_khop():
    rows = rbf_svm_study(params=((1, 0.01), (1000, 10)))
    good, over = rows
    assert over["train"] == 1.0 and over["test"] < good["test"]
