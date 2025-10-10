import math


class Vector2D(object):

    def __init__(self, x, y):
        """x, y坐标初始化向量"""
        self.x = x
        self.y = y
    def __add__(self, other):
        # +
        return Vector2D(self.x + other.x, self.y + other.y)
    def __sub__(self, other):
        # -
        return Vector2D(self.x - other.x, self.y - other.y)
    def __mul__(self, other):
        # x
        return self.x * other.x + self.y * other.y
    def __abs__(self):
        # ||
        return math.sqrt(self.x ** 2 + self.y ** 2)

    # __str__
    def __str__(self):
        return f"({self.x}, {self.y})"

    # 返回两个向量的夹角
    def angle(self, other):
        # 计算数量积
        dot_product = self * other  # 利用重载的 __mul__ 方法
        # 计算模长乘积
        magnitude_product = abs(self) * abs(other)  # 利用重载的 __abs__ 方法
        # 避免除以零的错误
        if magnitude_product == 0:
            return 0.0
        cos_theta = dot_product / magnitude_product
        # 使用 max/min 确保
        cos_theta = max(-1.0, min(1.0, cos_theta))

        # 返回夹角
        return math.acos(cos_theta)
a = Vector2D(1, 2)
b = Vector2D(2, 3)
print(f"a = {a}, b = {b}")
print(f"a + b = {a + b}")
print(f"a * b (点积) = {a * b}")
print(f"b - a = {b - a}")
print(f"|a| (模长) = {abs(a)}")
print(f"str(b) = {str(b)}")
print(f"a.angle(b) (弧度) = {a.angle(b)}")