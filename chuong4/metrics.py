"""Các độ đo đánh giá mô hình dùng cho bài 4.27 (hồi quy) và 4.28 (phân lớp)."""
import numpy as np


def r2_pearson(y_true, y_pred):
    """R2 theo nghĩa bình phương hệ số tương quan Pearson giữa y thật và y dự đoán.

    R2 = [ sum (y - ȳ)(ŷ - ŷ_mean) ]^2 / [ sum (y - ȳ)^2 * sum (ŷ - ŷ_mean)^2 ]
    Đo mức độ "cùng xu hướng" (tuyến tính) giữa dự đoán và thực tế, nằm trong [0, 1].
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    r = np.corrcoef(y_true, y_pred)[0, 1]
    return float(r ** 2)


def nse(y_true, y_pred):
    """Hệ số Nash-Sutcliffe Efficiency: NSE = 1 - sum (y - ŷ)^2 / sum (y - ȳ)^2.

    NSE = 1: dự đoán hoàn hảo; NSE = 0: tốt ngang việc đoán giá trị trung bình;
    NSE < 0: tệ hơn đoán trung bình. (Trùng công thức với `r2_score` của sklearn.)
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))


def mae(y_true, y_pred):
    """Mean Absolute Error: trung bình trị tuyệt đối sai số, cùng đơn vị với y."""
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def rmse(y_true, y_pred):
    """Root Mean Squared Error: căn của MSE, phạt nặng các sai số lớn."""
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


def regression_report(y_true, y_pred):
    """Gom 4 độ đo hồi quy (R2, NSE, MAE, RMSE) vào một dict."""
    return {"R2": r2_pearson(y_true, y_pred), "NSE": nse(y_true, y_pred),
            "MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred)}


def classification_report_binary(y_true, y_pred, positive=1):
    """Tính Accuracy, Precision, Recall, F1 cho bài toán 2 lớp từ ma trận nhầm lẫn.

    TP: dự đoán dương & đúng; FP: dự đoán dương & sai; FN: dự đoán âm & sai.
    Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2PR/(P+R).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_pred == positive) & (y_true == positive))
    fp = np.sum((y_pred == positive) & (y_true != positive))
    fn = np.sum((y_pred != positive) & (y_true == positive))
    acc = np.mean(y_true == y_pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"Accuracy": float(acc), "Precision": float(p), "Recall": float(r), "F1": float(f1)}
