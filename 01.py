def hillClimbing_best_direction(f, x, y, z, h=0.01):
    while True:
        current_value = f(x, y, z)
        print(f"x = {x:.3f}, y = {y:.3f}, z = {z:.3f}, f(x, y, z) = {current_value:.6f}")

        # 六個方向的候選點與對應的函數值
        candidates = [
            (x + h, y, z),
            (x - h, y, z),
            (x, y + h, z),
            (x, y - h, z),
            (x, y, z + h),
            (x, y, z - h)
        ]

        # 計算每個候選點的函數值
        evaluated = [(cx, cy, cz, f(cx, cy, cz)) for (cx, cy, cz) in candidates]

        # 找出使 f 最小的方向
        best = min(evaluated, key=lambda item: item[3])

        # 如果沒有更好的點，就停止
        if best[3] >= current_value:
            break
        else:
            x, y, z = best[0], best[1], best[2]

    print("\n最小點：")
    print(f"x = {x:.6f}, y = {y:.6f}, z = {z:.6f}")
    print(f"最小值 f(x, y, z) = {f(x, y, z):.12f}")
    return x, y, z, f(x, y, z)

# 目標函數
def f(x, y, z):
    return x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

# 執行
hillClimbing_best_direction(f, 0, 0, 0)
