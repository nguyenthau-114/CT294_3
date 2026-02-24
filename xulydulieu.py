#bai nhom
import pandas as pd                 # đọc/ xử lý dữ liệu
import joblib                      # lưu/load dữ liệu đã split
from sklearn.model_selection import train_test_split  # chia train/test
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix  # đánh giá

# 1) Đọc file + gán tên cột
def load_letter_data(path="letter-recognition.data"):
    cols = ["letter"] + [f"x{i}" for i in range(1, 17)]  # nhãn + 16 đặc trưng
    df = pd.read_csv(path, header=None, names=cols)      # file không có header
    return df

# 2) Làm sạch tối thiểu
def clean_data(df: pd.DataFrame):
    df = df.drop_duplicates().reset_index(drop=True)     # xoá dòng trùng (nếu có)
    return df

# 3) Tách X/y và chia train/test cố định
def make_splits(df: pd.DataFrame, test_size=0.2, seed=42):
    X = df.drop(columns=["letter"])                      # đặc trưng
    y = df["letter"]                                     # nhãn A-Z
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y  # giữ cân bằng lớp
    )
    labels = sorted(y.unique())                          # cố định thứ tự lớp
    return X_train, X_test, y_train, y_test, labels

# 4) Hàm đánh giá chung cho mọi mô hình
def evaluate(model, X_test, y_test, labels, name="Model"):
    pred = model.predict(X_test)                         # dự đoán
    acc = accuracy_score(y_test, pred)                   # accuracy
    f1m = f1_score(y_test, pred, average="macro")        # macro F1 (công bằng giữa các lớp)
    cm = confusion_matrix(y_test, pred, labels=labels)   # ma trận nhầm lẫn
    print(f"{name} | Accuracy={acc:.4f} | Macro-F1={f1m:.4f}")
    print(classification_report(y_test, pred))           # report chi tiết
    return acc, f1m, cm

# 5) Chạy file này để tạo dataset đã split dùng chung cho cả nhóm
if __name__ == "__main__":
    df = load_letter_data()                              # đọc dữ liệu
    df = clean_data(df)                                  # làm sạch tối thiểu

    print("Shape:", df.shape)                            # kích thước dữ liệu
    print("Missing:", df.isna().sum().sum())             # tổng số NaN
    print("Duplicates:", df.duplicated().sum())          # số dòng trùng

    X_train, X_test, y_train, y_test, labels = make_splits(df)  # chia train/test

    joblib.dump(                                         # lưu để nhóm DT/RF dùng chung
        {"X_train": X_train, "X_test": X_test,
         "y_train": y_train, "y_test": y_test,
         "labels": labels},
        "letter_shared_splits.pkl"
    )
    print("Saved: letter_shared_splits.pkl")             # báo đã lưu xong