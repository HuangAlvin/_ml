import random
import math

# 隨機產生城市座標
def generate_cities(n, width=1000, height=1000):
    return [(random.randint(0, width), random.randint(0, height)) for _ in range(n)]

# 計算兩點距離
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# 計算整條路徑的總距離
def total_distance(cities, path):
    return sum(distance(cities[path[i]], cities[path[(i + 1) % len(path)]]) for i in range(len(path)))

# 產生鄰近解：交換兩個城市的位置
def neighbor(path):
    new_path = path[:]
    i, j = random.sample(range(len(path)), 2)
    new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path

# 爬山演算法主程式
def hill_climb(cities, max_iterations=10000):
    current = list(range(len(cities)))
    random.shuffle(current)
    current_distance = total_distance(cities, current)

    for _ in range(max_iterations):
        candidate = neighbor(current)
        candidate_distance = total_distance(cities, candidate)
        if candidate_distance < current_distance:
            current, current_distance = candidate, candidate_distance

    return current, current_distance

# 執行
if __name__ == "__main__":
    city_count = 20
    cities = generate_cities(city_count)
    best_path, best_distance = hill_climb(cities)

    print("最佳路徑：", best_path)
    print("總距離：", best_distance)
