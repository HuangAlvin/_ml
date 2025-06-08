import torch

# 初始化參數，requires_grad=True 表示要追蹤梯度
x = torch.tensor([0.0], requires_grad=True)
y = torch.tensor([0.0], requires_grad=True)
z = torch.tensor([0.0], requires_grad=True)

# 學習率與訓練輪數
lr = 0.1
epochs = 100

for epoch in range(epochs):
    # 前向傳播：計算目標函數
    f = x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

    # 反向傳播：計算梯度
    f.backward()

    # 使用梯度更新參數
    with torch.no_grad():
        x -= lr * x.grad
        y -= lr * y.grad
        z -= lr * z.grad

        # 清除上一輪的梯度
        x.grad.zero_()
        y.grad.zero_()
        z.grad.zero_()

    # 印出每輪資訊（可選）
    print(f"Epoch {epoch+1:3d}: f = {f.item():.6f}, x = {x.item():.4f}, y = {y.item():.4f}, z = {z.item():.4f}")

# 最終結果
print(f"\nMinimum at x={x.item():.4f}, y={y.item():.4f}, z={z.item():.4f}")
print(f"Minimum value = {f.item():.6f}")
