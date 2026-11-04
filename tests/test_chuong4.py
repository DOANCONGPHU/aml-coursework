"""Kiểm thử Chương 4: tính tay forward/backward và lớp MLP tự cài đặt."""
import numpy as np
import pytest

from chuong4.mlp_numpy import MLP, softmax
from chuong4.tinh_tay import backward, bai_toan_mau


def test_cau_4_19_va_4_24_forward():
    """Câu 4.19/4.24: z1=[1,2], a1=[1,2], z2=3, đầu ra = 3 (đáp án C)."""
    x, y, W1, W2 = bai_toan_mau()
    cache, _, _, _ = backward(x, y, W1, W2)
    np.testing.assert_allclose(cache["z1"].ravel(), [1, 2])
    np.testing.assert_allclose(cache["a1"].ravel(), [1, 2])
    assert cache["z2"].item() == pytest.approx(3.0)
    assert cache["a2"].item() == pytest.approx(3.0)


def test_cau_4_20_gradient_W2():
    """Câu 4.20: dJ/dW2 = [-2, -4] (đáp án D)."""
    x, y, W1, W2 = bai_toan_mau()
    _, _, dW2, loss = backward(x, y, W1, W2)
    np.testing.assert_allclose(dW2, [[-2.0, -4.0]])
    assert loss == pytest.approx(2.0)


def test_cau_4_25_gradient_W1():
    """Câu 4.25: dJ/dW1 = [[-2,-4],[-2,-4]]."""
    x, y, W1, W2 = bai_toan_mau()
    _, dW1, _, _ = backward(x, y, W1, W2)
    np.testing.assert_allclose(dW1, [[-2.0, -4.0], [-2.0, -4.0]])


def test_cau_4_12_so_trong_so():
    """Câu 4.12: 3 đầu vào x 5 nơ-ron = 15 trọng số (không tính bias)."""
    m = MLP(hidden_layers=(5,), task="regression", epochs=1)
    m.fit(np.zeros((4, 3)), np.zeros(4))
    assert m.W[0].size == 15


def test_softmax_tong_bang_1():
    P = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]]))
    np.testing.assert_allclose(P.sum(axis=1), [1.0, 1.0])
    assert np.all(np.isfinite(P))


def _numerical_grad(model, X, Y, l, i, j, eps=1e-6):
    """Gradient số (sai phân trung tâm) của loss theo W[l][i, j]."""
    old = model.W[l][i, j]
    model.W[l][i, j] = old + eps
    lp = model._loss(Y, model._forward(X)[1][-1])
    model.W[l][i, j] = old - eps
    lm = model._loss(Y, model._forward(X)[1][-1])
    model.W[l][i, j] = old
    return (lp - lm) / (2 * eps)


@pytest.mark.parametrize("task,y_kind", [
    ("regression", "real"), ("classification", "binary"), ("classification", "multi")])
def test_backprop_khop_gradient_so(task, y_kind):
    """Gradient do backprop tính phải khớp gradient số -> backprop cài đúng."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 3))
    y = {"real": rng.normal(size=10),
         "binary": rng.integers(0, 2, 10),
         "multi": rng.integers(0, 3, 10)}[y_kind]
    m = MLP(hidden_layers=(4, 3), task=task, activation="tanh", l2=0.1, epochs=0)
    Y = m._prepare_targets(y)
    m._init_params(3, Y.shape[1])
    Zs, As = m._forward(X)
    gW, _ = m._backward(Zs, As, Y)
    for l in range(len(m.W)):
        for (i, j) in [(0, 0), (m.W[l].shape[0] - 1, m.W[l].shape[1] - 1)]:
            assert gW[l][i, j] == pytest.approx(_numerical_grad(m, X, Y, l, i, j), rel=1e-4, abs=1e-7)


def test_mlp_hoc_duoc_xor():
    """XOR là bài toán phi tuyến: MLP có tầng ẩn phải học được 100%."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    m = MLP(hidden_layers=(8,), task="classification", activation="tanh",
            lr=0.5, epochs=2000, batch_size=None, random_state=1).fit(X, y)
    assert (m.predict(X) == y).all()
    assert m.loss_history_[-1] < m.loss_history_[0]


def test_mlp_hoi_quy_va_da_lop():
    rng = np.random.default_rng(42)
    X = rng.uniform(-2, 2, size=(200, 2))
    y_reg = X[:, 0] ** 2 + X[:, 1]
    reg = MLP(hidden_layers=(32, 16), lr=0.01, epochs=300, random_state=0).fit(X, y_reg)
    mse = np.mean((reg.predict(X) - y_reg) ** 2)
    assert mse < 0.1 * np.var(y_reg)

    y_cls = np.array(["a", "b", "c"])[(X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int)]
    clf = MLP(hidden_layers=(16,), task="classification", lr=0.1, epochs=300).fit(X, y_cls)
    assert clf.predict_proba(X).shape == (200, 3)
    assert np.mean(clf.predict(X) == y_cls) > 0.9


def test_tham_so_sai_bao_loi():
    with pytest.raises(ValueError):
        MLP(task="clustering")
    with pytest.raises(ValueError):
        MLP(activation="abc")
