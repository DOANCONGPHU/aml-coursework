"""Bài 4.26 - Cài đặt từ đầu (chỉ dùng NumPy) quá trình huấn luyện mạng nơ-ron nhiều tầng.

- Số tầng ẩn và số nơ-ron mỗi tầng ẩn: tùy ý, truyền qua tham số `hidden_layers`.
- Số nơ-ron tầng ra: tùy ý, tự suy ra từ dữ liệu y và loại bài toán:
    * task="regression"      -> tầng ra tuyến tính, hàm mất mát MSE (y có thể nhiều cột).
    * task="classification" -> 2 lớp: 1 nơ-ron sigmoid + Binary Cross-Entropy;
                               K>2 lớp: K nơ-ron softmax + Cross-Entropy.
- Tối ưu bằng mini-batch Gradient Descent, gradient tính bằng Backpropagation.
"""
import numpy as np


# ----------------------------------------------------------------------------------
# Các hàm kích hoạt và đạo hàm tương ứng
# ----------------------------------------------------------------------------------
def relu(z):
    """ReLU(z) = max(0, z). Hàm kích hoạt phi tuyến phổ biến cho tầng ẩn."""
    return np.maximum(0.0, z)


def relu_grad(z):
    """Đạo hàm ReLU: 1 nếu z > 0, ngược lại 0."""
    return (z > 0).astype(float)


def tanh(z):
    """tanh(z): đầu ra thuộc (-1, 1), đối xứng quanh 0."""
    return np.tanh(z)


def tanh_grad(z):
    """Đạo hàm tanh: 1 - tanh(z)^2."""
    return 1.0 - np.tanh(z) ** 2


def sigmoid(z):
    """Sigmoid ổn định số học: 1 / (1 + e^{-z}), cắt z để tránh tràn số exp."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_grad(z):
    """Đạo hàm sigmoid: s(z) * (1 - s(z))."""
    s = sigmoid(z)
    return s * (1.0 - s)


def softmax(z):
    """Softmax theo từng hàng; trừ max mỗi hàng để ổn định số học."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


ACTIVATIONS = {
    "relu": (relu, relu_grad),
    "tanh": (tanh, tanh_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
}


class MLP:
    """Mạng nơ-ron nhiều tầng (Multilayer Perceptron) huấn luyện bằng backpropagation.

    Tham số
    -------
    hidden_layers : tuple[int]  số nơ-ron của từng tầng ẩn, ví dụ (16, 8) = 2 tầng ẩn.
    task          : "regression" hoặc "classification".
    activation    : hàm kích hoạt tầng ẩn ("relu", "tanh", "sigmoid").
    lr            : learning rate (bước nhảy của gradient descent).
    epochs        : số lần duyệt toàn bộ dữ liệu.
    batch_size    : kích thước mini-batch (None = full-batch GD).
    l2            : hệ số điều chuẩn L2 (weight decay) để giảm quá khớp.
    random_state  : hạt giống ngẫu nhiên để kết quả lặp lại được.
    """

    def __init__(self, hidden_layers=(16,), task="regression", activation="relu",
                 lr=0.01, epochs=500, batch_size=32, l2=0.0, random_state=0, verbose=False):
        if task not in ("regression", "classification"):
            raise ValueError("task phải là 'regression' hoặc 'classification'")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation phải thuộc {list(ACTIVATIONS)}")
        self.hidden_layers = tuple(hidden_layers)
        self.task = task
        self.activation = activation
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------------------
    # Khởi tạo
    # ------------------------------------------------------------------------------
    def _init_params(self, n_in, n_out):
        """Khởi tạo trọng số W và bias b cho mọi tầng.

        Dùng khởi tạo He (cho ReLU) hoặc Xavier (cho tanh/sigmoid) để phương sai tín
        hiệu không bị bùng nổ/triệt tiêu khi đi qua nhiều tầng. Bias khởi tạo bằng 0.
        """
        rng = np.random.default_rng(self.random_state)
        sizes = [n_in, *self.hidden_layers, n_out]
        self.W, self.b = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            scale = np.sqrt(2.0 / fan_in) if self.activation == "relu" else np.sqrt(1.0 / fan_in)
            self.W.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            self.b.append(np.zeros((1, fan_out)))

    def _prepare_targets(self, y):
        """Chuẩn bị nhãn y thành ma trận mục tiêu Y và xác định số nơ-ron đầu ra.

        - Hồi quy: Y có shape (n, k) với k = số biến mục tiêu.
        - Phân lớp 2 lớp: Y shape (n, 1) gồm 0/1, dùng sigmoid.
        - Phân lớp K lớp: Y one-hot shape (n, K), dùng softmax.
        """
        y = np.asarray(y)
        if self.task == "regression":
            Y = y.reshape(len(y), -1).astype(float)
            self.output_ = "linear"
            return Y
        self.classes_ = np.unique(y)
        idx = np.searchsorted(self.classes_, y)
        if len(self.classes_) == 2:
            self.output_ = "sigmoid"
            return idx.reshape(-1, 1).astype(float)
        self.output_ = "softmax"
        Y = np.zeros((len(y), len(self.classes_)))
        Y[np.arange(len(y)), idx] = 1.0
        return Y

    # ------------------------------------------------------------------------------
    # Lan truyền xuôi / ngược
    # ------------------------------------------------------------------------------
    def _forward(self, X):
        """Lan truyền xuôi (forward propagation).

        Với mỗi tầng l: Z_l = A_{l-1} W_l + b_l ; A_l = f(Z_l).
        Tầng cuối dùng hàm ra tương ứng bài toán (tuyến tính / sigmoid / softmax).
        Trả về danh sách Z và A của mọi tầng (A[0] = X) để dùng khi lan truyền ngược.
        """
        act, _ = ACTIVATIONS[self.activation]
        Zs, As = [], [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z = As[-1] @ W + b
            Zs.append(Z)
            if i < len(self.W) - 1:
                As.append(act(Z))
            elif self.output_ == "sigmoid":
                As.append(sigmoid(Z))
            elif self.output_ == "softmax":
                As.append(softmax(Z))
            else:
                As.append(Z)
        return Zs, As

    def _loss(self, Y, Y_hat):
        """Tính hàm mất mát trung bình trên batch (+ phạt L2 nếu có).

        - MSE           : 1/(2n) * sum (Y - Y_hat)^2
        - Binary CE     : -1/n * sum [y log p + (1-y) log(1-p)]
        - Categorical CE: -1/n * sum y log p
        """
        n = Y.shape[0]
        eps = 1e-12
        if self.output_ == "linear":
            loss = 0.5 * np.sum((Y - Y_hat) ** 2) / n
        elif self.output_ == "sigmoid":
            P = np.clip(Y_hat, eps, 1 - eps)
            loss = -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P)) / n
        else:
            loss = -np.sum(Y * np.log(np.clip(Y_hat, eps, 1.0))) / n
        if self.l2:
            loss += 0.5 * self.l2 * sum(np.sum(W ** 2) for W in self.W) / n
        return float(loss)

    def _backward(self, Zs, As, Y):
        """Lan truyền ngược (backpropagation) - tính gradient của loss theo W, b.

        Với cặp (MSE, tuyến tính), (BCE, sigmoid), (CE, softmax) thì sai số ở tầng ra
        đều rút gọn về cùng một dạng: delta_L = (Y_hat - Y) / n.
        Sau đó lan truyền sai số từ tầng ra về tầng vào (Output -> Input):
            dW_l = A_{l-1}^T delta_l ;  db_l = sum(delta_l)
            delta_{l-1} = (delta_l W_l^T) ⊙ f'(Z_{l-1})
        """
        _, act_grad = ACTIVATIONS[self.activation]
        n = Y.shape[0]
        delta = (As[-1] - Y) / n
        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)
        for l in range(len(self.W) - 1, -1, -1):
            grads_W[l] = As[l].T @ delta + self.l2 * self.W[l] / n
            grads_b[l] = delta.sum(axis=0, keepdims=True)
            if l > 0:
                delta = (delta @ self.W[l].T) * act_grad(Zs[l - 1])
        return grads_W, grads_b

    # ------------------------------------------------------------------------------
    # API công khai
    # ------------------------------------------------------------------------------
    def fit(self, X, y):
        """Huấn luyện mạng trên (X, y) bằng mini-batch gradient descent.

        Mỗi epoch: xáo trộn dữ liệu, chia mini-batch, với mỗi batch thực hiện
        forward -> backward -> cập nhật W := W - lr * dW. Lưu loss mỗi epoch
        vào `self.loss_history_` để vẽ đường cong huấn luyện.
        """
        X = np.asarray(X, dtype=float)
        Y = self._prepare_targets(y)
        self._init_params(X.shape[1], Y.shape[1])
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        bs = n if self.batch_size is None else min(self.batch_size, n)
        self.loss_history_ = []
        for epoch in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                Zs, As = self._forward(X[idx])
                gW, gb = self._backward(Zs, As, Y[idx])
                for l in range(len(self.W)):
                    self.W[l] -= self.lr * gW[l]
                    self.b[l] -= self.lr * gb[l]
            _, As = self._forward(X)
            self.loss_history_.append(self._loss(Y, As[-1]))
            if self.verbose and (epoch % max(1, self.epochs // 10) == 0):
                print(f"epoch {epoch:4d}  loss = {self.loss_history_[-1]:.5f}")
        return self

    def predict_proba(self, X):
        """Trả về xác suất từng lớp (chỉ dùng cho phân lớp), shape (n, số lớp)."""
        if self.task != "classification":
            raise AttributeError("predict_proba chỉ dùng cho bài toán phân lớp")
        _, As = self._forward(np.asarray(X, dtype=float))
        P = As[-1]
        if self.output_ == "sigmoid":
            P = np.hstack([1 - P, P])
        return P

    def predict(self, X):
        """Dự đoán: hồi quy trả về giá trị thực; phân lớp trả về nhãn lớp gốc."""
        X = np.asarray(X, dtype=float)
        if self.task == "regression":
            out = self._forward(X)[1][-1]
            return out.ravel() if out.shape[1] == 1 else out
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


if __name__ == "__main__":
    # Demo nhỏ: học hàm XOR - bài toán phi tuyến mà perceptron 1 tầng không giải được.
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    model = MLP(hidden_layers=(8,), task="classification", activation="tanh",
                lr=0.5, epochs=2000, batch_size=None, random_state=1).fit(X, y)
    print("Dự đoán XOR:", model.predict(X), "| loss cuối:", round(model.loss_history_[-1], 4))
