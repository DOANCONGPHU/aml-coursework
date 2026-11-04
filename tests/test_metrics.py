"""Kiểm thử các độ đo tự cài đặt (chuong4/metrics.py) bằng cách đối chiếu với sklearn."""
import numpy as np
import pytest
from sklearn import metrics as skm

from chuong4.metrics import classification_report_binary, mae, nse, r2_pearson, rmse


def test_do_do_hoi_quy_khop_sklearn():
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    p = y + rng.normal(scale=0.5, size=50)
    assert nse(y, p) == pytest.approx(skm.r2_score(y, p))
    assert mae(y, p) == pytest.approx(skm.mean_absolute_error(y, p))
    assert rmse(y, p) == pytest.approx(np.sqrt(skm.mean_squared_error(y, p)))
    assert r2_pearson(y, p) == pytest.approx(np.corrcoef(y, p)[0, 1] ** 2)


def test_r2_pearson_khac_nse_khi_lech_he_thong():
    """Dự đoán lệch một hằng số: R2 (Pearson) vẫn = 1 nhưng NSE giảm mạnh."""
    y = np.arange(10.0)
    p = y + 5
    assert r2_pearson(y, p) == pytest.approx(1.0)
    assert nse(y, p) < 0


def test_do_do_phan_lop_khop_sklearn():
    y = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    p = np.array([1, 0, 0, 1, 1, 1, 0, 0, 1, 0])
    r = classification_report_binary(y, p)
    assert r["Accuracy"] == pytest.approx(skm.accuracy_score(y, p))
    assert r["Precision"] == pytest.approx(skm.precision_score(y, p))
    assert r["Recall"] == pytest.approx(skm.recall_score(y, p))
    assert r["F1"] == pytest.approx(skm.f1_score(y, p))
