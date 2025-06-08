# iris_model_demo.py

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
import os

# === 第一步：準備資料與訓練模型 ===
iris = load_iris()
X, y = iris.data, iris.target

# 分割訓練與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立並訓練模型
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 測試準確率
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型訓練完成，測試準確率：{accuracy:.2f}")

# === 第二步：儲存模型到檔案 ===
model_filename = "iris_model.pkl"
joblib.dump(model, model_filename)
print(f"模型已儲存為 '{model_filename}'")

# === 第三步：載入模型並進行預測 ===
if os.path.exists(model_filename):
    loaded_model = joblib.load(model_filename)
    # 測試用資料（範例：setosa）
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    predicted_class = loaded_model.predict(sample)
    class_name = iris.target_names[predicted_class[0]]
    print(f"預測結果：類別編號 {predicted_class[0]}，名稱為 '{class_name}'")
else:
    print("找不到模型檔案，請先訓練並儲存模型。")
