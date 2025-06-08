from micrograd.engine import Value

# 初始化變數
x = Value(0.0, label='x')
y = Value(0.0, label='y')
z = Value(0.0, label='z')

# 設定學習率
learning_rate = 0.1

# 執行梯度下降
for step in range(100):
    # 前向傳播：定義損失函數 f = x² + y² + z² - 2x - 4y - 6z + 8
    f = x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8
    f.label = 'f'

    # 清空梯度
    x.grad = y.grad = z.grad = 0.0

    # 反向傳播
    f.backward()

    # 更新參數
    x.data -= learning_rate * x.grad
    y.data -= learning_rate * y.grad
    z.data -= learning_rate * z.grad

    # 每10步輸出一次
    if step % 10 == 0:
        print(f"step {step:3d}: f = {f.data:.4f}, x = {x.data:.4f}, y = {y.data:.4f}, z = {z.data:.4f}")
