import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 建立訓練資料（y = 2x + 1）
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_train = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

# 初始化權重與偏差（需設定 requires_grad=True 以便計算梯度）
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# 學習率
lr = 0.01

# 訓練迴圈
for epoch in range(1000):
    # 預測值 y_pred = x * w + b
    y_pred = x_train * w + b

    # 計算均方誤差
    loss = F.mse_loss(y_pred, y_train)

    # 反向傳播計算梯度
    loss.backward()

    # 更新權重與偏差（手動梯度下降）
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

        # 清空梯度以避免累積
        w.grad.zero_()
        b.grad.zero_()

    # 每 100 次列印一次損失與參數
    if epoch % 100 == 0:
        print(f'Epoch {epoch}: loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}')

# 測試與可視化
x_test = torch.linspace(0, 5, 100).unsqueeze(1)
y_test = x_test * w.detach() + b.detach()

plt.scatter(x_train, y_train, label='Training data')
plt.plot(x_test, y_test, 'r', label='Fitted line')
plt.legend()
plt.title('Linear Regression using PyTorch')
plt.show()
