import numpy as np
from PIL import Image
import os
import sys

# 请确保您的 'bird.jpg' 文件位于此路径下
INPUT_PATH = 'bird.jpg'
OUTPUT_DIR = 'Output'


def process_image():
    try:
        # 1. 读取图像并转换为 NumPy 数组
        img = Image.open(INPUT_PATH).convert('RGB')
        original_arr = np.array(img)

        # 确保输出目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # 对 ndarray 对象执行 sort 函数
        # 对最后一个轴（颜色通道：R, G, B）进行排序，这样每个像素的三个颜色分量值会被重新排列。
        arr_sort = np.sort(original_arr, axis=2)
        img_sort = Image.fromarray(arr_sort.astype(np.uint8))
        save_path_1 = os.path.join(OUTPUT_DIR, '1.jpg')
        img_sort.save(save_path_1)
        print(f"\n1已完成，保存至: {save_path_1}")

        # 对 ndarray 对象执行 transpose((0, 1, 2))
        # (0, 1, 2) 对应于 (高度, 宽度, 颜色通道)
        arr_transpose = original_arr.transpose((0, 1, 2))
        img_transpose = Image.fromarray(arr_transpose.astype(np.uint8))
        save_path_2 = os.path.join(OUTPUT_DIR, '2.jpg')
        img_transpose.save(save_path_2)
        print(f"\n2已完成，保存至: {save_path_2}")

        # 使用 255 减去 ndarray 对象
        # NumPy 允许广播操作，255 会减去数组中的每个元素
        arr_negative = 255 - original_arr
        img_negative = Image.fromarray(arr_negative.astype(np.uint8))
        save_path_3 = os.path.join(OUTPUT_DIR, '3.jpg')
        img_negative.save(save_path_3)

        print(f"\n3已完成，保存至: {save_path_3}")
    except FileNotFoundError:
        sys.exit(f"错误：文件操作失败，请检查路径和权限。")
    except Exception as e:
        sys.exit(f"处理图像时发生错误: {e}")
if __name__ == "__main__":
    process_image()
