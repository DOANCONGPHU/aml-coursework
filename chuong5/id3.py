"""Bài 5.22 - Cài đặt thuật toán ID3 bằng Python thuần (không dùng sklearn).

Cây được biểu diễn dưới dạng dictionary lồng nhau:
    {"Tuổi": {"Trẻ": {"Sinh viên": {"Có": "Có", "Không": "Không"}},
              "Trung niên": "Có",
              "Già": {...}}}
- Khóa ngoài cùng là tên thuộc tính được chọn để phân nhánh.
- Mỗi giá trị của thuộc tính trỏ tới hoặc một cây con (dict) hoặc một nhãn lá (str).
"""
from collections import Counter
from math import log2

# Bảng dữ liệu "Mua máy tính" trong đề bài 5.22 (bỏ cột ID vì không mang thông tin).
ATTRIBUTES = ["Tuổi", "Thu nhập", "Sinh viên", "Đánh giá tín dụng"]
TARGET = "Mua máy tính"
_ROWS = [
    ("Trẻ", "Cao", "Không", "Tốt", "Không"),
    ("Trẻ", "Cao", "Không", "Xuất sắc", "Không"),
    ("Trung niên", "Cao", "Không", "Tốt", "Có"),
    ("Già", "Trung bình", "Không", "Tốt", "Có"),
    ("Già", "Thấp", "Có", "Tốt", "Có"),
    ("Già", "Thấp", "Có", "Xuất sắc", "Không"),
    ("Trung niên", "Thấp", "Có", "Xuất sắc", "Có"),
    ("Trẻ", "Trung bình", "Không", "Tốt", "Không"),
    ("Trẻ", "Thấp", "Có", "Tốt", "Có"),
    ("Già", "Trung bình", "Có", "Tốt", "Có"),
    ("Trẻ", "Trung bình", "Có", "Xuất sắc", "Có"),
    ("Trung niên", "Trung bình", "Không", "Xuất sắc", "Có"),
    ("Trung niên", "Cao", "Có", "Tốt", "Có"),
    ("Già", "Trung bình", "Không", "Xuất sắc", "Không"),
]
DATA = [dict(zip(ATTRIBUTES + [TARGET], r)) for r in _ROWS]


def entropy(labels):
    """Entropy Shannon (đơn vị bit) của một danh sách nhãn:
    H(S) = - sum_c p_c * log2(p_c), với p_c là tỉ lệ mẫu thuộc lớp c.
    H = 0 khi tập thuần nhất; H lớn nhất khi các lớp phân bố đều.
    """
    n = len(labels)
    if n == 0:
        return 0.0
    return -sum((c / n) * log2(c / n) for c in Counter(labels).values())


def split(data, attribute):
    """Chia tập dữ liệu theo từng giá trị của `attribute`.
    Trả về dict {giá trị: danh sách các mẫu có giá trị đó}."""
    groups = {}
    for row in data:
        groups.setdefault(row[attribute], []).append(row)
    return groups


def information_gain(data, attribute, target=TARGET):
    """Information Gain của thuộc tính A trên tập S:
        Gain(S, A) = H(S) - sum_v |S_v|/|S| * H(S_v)
    tức là mức giảm entropy (độ hỗn loạn) sau khi phân nhánh theo A.
    """
    base = entropy([r[target] for r in data])
    n = len(data)
    remainder = sum(len(sub) / n * entropy([r[target] for r in sub])
                    for sub in split(data, attribute).values())
    return base - remainder


def majority_label(data, target=TARGET):
    """Nhãn chiếm đa số trong tập dữ liệu (dùng làm lá khi không thể chia tiếp).
    Nếu hòa, chọn nhãn xuất hiện trước trong dữ liệu để kết quả ổn định."""
    return Counter(r[target] for r in data).most_common(1)[0][0]


def build_tree(data, attributes=None, target=TARGET):
    """Xây cây quyết định bằng ID3 (đệ quy), trả về dictionary lồng nhau.

    Điều kiện dừng (tạo nút lá):
      1. Mọi mẫu trong nút cùng một lớp -> lá mang nhãn đó.
      2. Hết thuộc tính để chia -> lá mang nhãn đa số.
    Ngược lại: chọn thuộc tính có Information Gain LỚN NHẤT, tạo một nhánh cho mỗi
    giá trị của thuộc tính đó, rồi đệ quy với các thuộc tính còn lại.
    """
    if attributes is None:
        attributes = [a for a in data[0] if a != target]
    labels = [r[target] for r in data]
    if len(set(labels)) == 1:
        return labels[0]
    if not attributes:
        return majority_label(data, target)

    best = max(attributes, key=lambda a: information_gain(data, a, target))
    remaining = [a for a in attributes if a != best]
    tree = {best: {}}
    for value, subset in split(data, best).items():
        tree[best][value] = build_tree(subset, remaining, target)
    # Lưu nhãn đa số của nút để xử lý giá trị chưa từng thấy khi dự đoán.
    tree["__default__"] = majority_label(data, target)
    return tree


def predict(tree, sample):
    """Dự đoán nhãn của một mẫu mới (dict thuộc tính -> giá trị) bằng cây ID3.

    Đi từ gốc xuống: tại mỗi nút đọc thuộc tính phân nhánh, lấy giá trị của mẫu và
    theo nhánh tương ứng cho đến khi gặp lá. Nếu giá trị chưa xuất hiện khi huấn luyện
    thì trả về nhãn đa số của nút hiện tại.
    """
    while isinstance(tree, dict):
        attribute = next(k for k in tree if k != "__default__")
        branches = tree[attribute]
        value = sample.get(attribute)
        if value not in branches:
            return tree["__default__"]
        tree = branches[value]
    return tree


def print_tree(tree, indent=""):
    """In cây ra màn hình dạng thụt lề cho dễ đọc."""
    if not isinstance(tree, dict):
        print(f"{indent}-> {tree}")
        return
    attribute = next(k for k in tree if k != "__default__")
    for value, sub in tree[attribute].items():
        if isinstance(sub, dict):
            print(f"{indent}[{attribute} = {value}]")
            print_tree(sub, indent + "    ")
        else:
            print(f"{indent}[{attribute} = {value}] -> {sub}")


if __name__ == "__main__":
    print(f"H(S) = {entropy([r[TARGET] for r in DATA]):.4f}")
    for a in ATTRIBUTES:
        print(f"Gain(S, {a}) = {information_gain(DATA, a):.4f}")
    tree = build_tree(DATA, ATTRIBUTES)
    print_tree(tree)
    tests = [
        {"Tuổi": "Trẻ", "Thu nhập": "Cao", "Sinh viên": "Có", "Đánh giá tín dụng": "Tốt"},
        {"Tuổi": "Già", "Thu nhập": "Thấp", "Sinh viên": "Không", "Đánh giá tín dụng": "Xuất sắc"},
        {"Tuổi": "Trung niên", "Thu nhập": "Thấp", "Sinh viên": "Không", "Đánh giá tín dụng": "Tốt"},
        {"Tuổi": "Trẻ", "Thu nhập": "Thấp", "Sinh viên": "Không", "Đánh giá tín dụng": "Xuất sắc"},
    ]
    for s in tests:
        print(s, "=>", predict(tree, s))
    acc = sum(predict(tree, r) == r[TARGET] for r in DATA) / len(DATA)
    print(f"Độ chính xác trên tập huấn luyện: {acc:.0%}")
