from PIL import Image, ImageFilter  # 导入 ImageFilter 模块
import numpy as np


def gaussian_blur(pixels, radius):
    # 1. 将 NumPy 数组转换回 PIL Image 对象
    img = Image.fromarray(pixels)

    # 2. 应用高斯模糊滤镜
    # radius 参数控制模糊的程度（即高斯核的标准差sigma）
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius=int(radius)))

    # 3. 将模糊后的 PIL Image 对象转换回 NumPy 数组
    # 确保数据类型与原始输入一致，通常是 np.uint8
    gaussian_bird = np.array(blurred_img)

    return gaussian_bird

# 读取图像
pixel_mat = np.array(Image.open("./bird.jpg"))

# 从用户获取半径输入
# 注意：input() 接收的是字符串，需要转换为整数
radius_str = input("请输入高斯模糊半径（整数，如 2, 5, 10）：")
try:
    radius = int(radius_str)
except ValueError:
    print("输入无效，将使用默认半径 2。")
    radius = 2

# 执行高斯模糊
pixel_mat = gaussian_blur(pixel_mat, radius)

# 将 NumPy 数组转换回 PIL Image 对象并显示/保存
img = Image.fromarray(pixel_mat)
img.show()
img.save("./gaussian_bird.jpg")