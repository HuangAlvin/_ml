import numpy as np

# 七段顯示器的目標輸出（0–9）
seven_seg = np.array([
    [1,1,1,1,1,1,0],
    [0,1,1,0,0,0,0],
    [1,1,0,1,1,0,1],
    [1,1,1,1,0,0,1],
    [0,1,1,0,0,1,1],
    [1,0,1,1,0,1,1],
    [1,0,1,1,1,1,1],
    [1,1,1,0,0,0,0],
    [1,1,1,1,1,1,1],
    [1,1,1,1,0,1,1]
])

# 輸入為 one-hot 編碼的 0~9
inputs = np.eye(10)

# 初始化權重（輸入層 -> 隱藏層 -> 輸出層）
np.random.seed(0)
hidden_size = 10
W1 = np.random.randn(10, hidden_size) * 0.1
W2 = np.random.randn(hidden_size, 7) * 0.1

# Sigmoid 函數及其導數
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 損失函數：平方誤差
def loss(W1, W2):
    z1 = sigmoid(inputs @ W1)
    z2 = sigmoid(z1 @ W2)
    return np.mean((z2 - seven_seg) ** 2)

# 數值梯度計算
def numerical_gradient(W1, W2, epsilon=1e-4):
    grad_W1 = np.zeros_like(W1)
    grad_W2 = np.zeros_like(W2)

    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_plus = W1.copy()
            W1_minus = W1.copy()
            W1_plus[i, j] += epsilon
            W1_minus[i, j] -= epsilon
            grad_W1[i, j] = (loss(W1_plus, W2) - loss(W1_minus, W2)) / (2 * epsilon)

    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2_plus = W2.copy()
            W2_minus = W2.copy()
            W2_plus[i, j] += epsilon
            W2_minus[i, j] -= epsilon
            grad_W2[i, j] = (loss(W1, W2_plus) - loss(W1, W2_minus)) / (2 * epsilon)

    return grad_W1, grad_W2

# 訓練
learning_rate = 1.0
for epoch in range(200):
    grad_W1, grad_W2 = numerical_gradient(W1, W2)
    W1 -= learning_rate * grad_W1
    W2 -= learning_rate * grad_W2

    if epoch % 20 == 0:
        current_loss = loss(W1, W2)
        print(f"Epoch {epoch}: Loss = {current_loss:.6f}")

# 驗證輸出
z1 = sigmoid(inputs @ W1)
z2 = sigmoid(z1 @ W2)
predictions = (z2 > 0.5).astype(int)
print("預測結果：")
print(predictions)
