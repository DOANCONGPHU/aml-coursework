"""Kiểm chứng bằng code các câu tính tay 4.19, 4.20, 4.24, 4.25 (Chương 4).

Mạng: 2 nơ-ron vào -> 1 tầng ẩn 2 nơ-ron (ReLU) -> 1 nơ-ron ra (ReLU), không bias.
    x = [1, 2]^T,  W1 = [[1, 0], [0, 1]],  W2 = [[1, 1]],  y = 5,  J = 1/2 (y - y_hat)^2
"""
import numpy as np


def relu(z):
    """Hàm kích hoạt ReLU: trả về max(0, z) theo từng phần tử."""
    return np.maximum(0.0, z)


def relu_grad(z):
    """Đạo hàm của ReLU theo z: 1 nếu z > 0, ngược lại 0 (dùng ở bước lan truyền ngược)."""
    return (z > 0).astype(float)


def forward(x, W1, W2):
    """Lan truyền xuôi: tính z1 = W1 x, a1 = ReLU(z1), z2 = W2 a1, a2 = ReLU(z2).

    Trả về dict chứa toàn bộ giá trị trung gian để phục vụ việc lan truyền ngược.
    """
    z1 = W1 @ x
    a1 = relu(z1)
    z2 = W2 @ a1
    a2 = relu(z2)
    return {"z1": z1, "a1": a1, "z2": z2, "a2": a2}


def backward(x, y, W1, W2):
    """Lan truyền ngược cho J = 1/2 (y - y_hat)^2.

    Các bước:
        dJ/da2  = -(y - a2)
        delta2  = dJ/da2 * ReLU'(z2)
        dJ/dW2  = delta2 a1^T
        delta1  = (W2^T delta2) ⊙ ReLU'(z1)
        dJ/dW1  = delta1 x^T
    Trả về (cache forward, dJ/dW1, dJ/dW2, giá trị loss).
    """
    cache = forward(x, W1, W2)
    y_hat = cache["a2"]
    loss = 0.5 * float(np.sum((y - y_hat) ** 2))
    delta2 = -(y - y_hat) * relu_grad(cache["z2"])
    dW2 = delta2 @ cache["a1"].T
    delta1 = (W2.T @ delta2) * relu_grad(cache["z1"])
    dW1 = delta1 @ x.T
    return cache, dW1, dW2, loss


def bai_toan_mau():
    """Trả về (x, y, W1, W2) đúng như đề bài 4.19/4.20/4.24/4.25 (vector cột)."""
    x = np.array([[1.0], [2.0]])
    W1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    W2 = np.array([[1.0, 1.0]])
    y = np.array([[5.0]])
    return x, y, W1, W2


if __name__ == "__main__":
    x, y, W1, W2 = bai_toan_mau()
    cache, dW1, dW2, loss = backward(x, y, W1, W2)
    for k, v in cache.items():
        print(f"{k} = {v.ravel()}")
    print("Loss J =", loss)
    print("dJ/dW2 =", dW2)
    print("dJ/dW1 =\n", dW1)
