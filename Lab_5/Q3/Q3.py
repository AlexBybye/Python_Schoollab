from PIL import Image
import os
import math

# --- 配置参数 ---
TARGET_IMAGE_PATH = 'bird.jpg'
OUTPUT_IMAGE_PATH = 'converted_bird..jpg'
TILE_SIZE = 16  # 小图的尺寸，即 16x16 像素
NUM_TILES = 6  # 小图总数

# 计算每个灰度区间的宽度 (255 / 6)
# math.ceil 确保我们在边界情况下不会溢出
GRAY_BIN_WIDTH = 256 / NUM_TILES


def load_tiles():
    """
    加载所有小图，按文件序号（1-6）存储，并计算它们的平均灰度值（仅供参考）。
    返回结构：[(文件序号, Image对象), ...]
    **注意：这里假设文件序号 1-6 对应从小到大/暗到亮（或反之）的亮度顺序。**
    """
    tiles_data = []

    for i in range(1, NUM_TILES + 1):
        file_name = f"{i}.jpg"
        file_path = os.path.join(file_name)

        try:
            tile_img = Image.open(file_path).resize((TILE_SIZE, TILE_SIZE))

            # 仅计算平均灰度值用于调试或记录，不再用于匹配
            gray_tile = tile_img.convert('L')
            avg_color = sum(gray_tile.getdata()) / (TILE_SIZE * TILE_SIZE)

            # 存储：(文件序号, Image对象)
            tiles_data.append((i, tile_img))
            # print(f"Loaded tile {i}.jpg with avg_gray: {avg_color:.2f}")

        except FileNotFoundError:
            print(f"Error: Tile image not found at {file_path}. Please check Q3 folder.")
            return None
        except Exception as e:
            print(f"Error loading tile {file_name}: {e}")
            return None

    # 按文件序号排序 (即 1, 2, 3, 4, 5, 6)
    return sorted(tiles_data, key=lambda item: item[0])


def find_best_tile(target_avg_gray, sorted_tiles):
    """
    根据目标平均灰度值，将其强制量化到 6 个预设的灰度区间之一。

    我们沿用上一次的修正思路：**反转亮度映射**，即亮区域（高灰度）匹配暗小图，暗区域（低灰度）匹配亮小图。
    """
    if not sorted_tiles:
        return None

    # --- 1. 亮度反转 (如果仍觉得是反的) ---
    # 如果目标区域越亮 (255)，我们希望匹配到序号越小（越暗）的小图。
    # 目标区域越暗 (0)，我们希望匹配到序号越大（越亮）的小图。
    target_value = 255 - target_avg_gray

    # 如果您不需要反转，请使用：target_value = target_avg_gray

    # --- 2. 强制分箱量化 ---
    # 强制将目标灰度值映射到 0 到 5 的索引 (对应小图 1 到 6)

    # 索引计算: target_value / 42.66 -> [0, 5.99]
    tile_index_raw = target_value / GRAY_BIN_WIDTH

    # 向下取整得到 0-5 的索引
    tile_index = math.floor(tile_index_raw)

    # 确保索引不越界 (防止 target_value = 255 导致 tile_index_raw 接近 6)
    tile_index = min(tile_index, NUM_TILES - 1)

    # 从 sorted_tiles (存储格式: (文件序号, Image对象)) 中取出 Image 对象
    # 列表索引 0 对应 文件序号 1，索引 5 对应 文件序号 6
    # 我们可以直接使用 tile_index 作为列表索引

    # print(f"Gray: {target_avg_gray:.0f} -> Inverted: {target_value:.0f} -> Index: {tile_index + 1}")

    # sorted_tiles 已经是按文件序号 1-6 排序的
    _, best_tile = sorted_tiles[tile_index]

    return best_tile


def create_photo_mosaic(sorted_tiles):
    """
    主函数：读取 bird.jpg，进行分块，计算灰度，替换小图，并保存。
    """
    if not sorted_tiles:
        print("Cannot create mosaic: Tiles not loaded.")
        return

    try:
        # 1. 打开目标图片
        target_img = Image.open(TARGET_IMAGE_PATH)

        # 创建一个同尺寸的新的 RGB 图像，用于粘贴小图
        new_img = Image.new('RGB', target_img.size)

        # 2. 将 bird.jpg 转为灰度图像，用于计算平均灰度
        target_gray = target_img.convert('L')

    except FileNotFoundError:
        print(f"Error: Target image not found at {TARGET_IMAGE_PATH}.")
        return

    width, height = target_img.size

    # 3. 遍历目标图片，划分小方块
    for y in range(0, height, TILE_SIZE):
        for x in range(0, width, TILE_SIZE):

            # 确定当前方块的边界
            box = (x, y, x + TILE_SIZE, y + TILE_SIZE)

            # 截取目标灰度图像中的一块区域
            block_gray = target_gray.crop(box)

            # 4. 计算当前方块的平均灰度值
            block_pixels = list(block_gray.getdata())
            if not block_pixels:
                continue

            avg_gray = sum(block_pixels) / len(block_pixels)

            # 5. 将灰度值对应到6张小图片
            best_tile = find_best_tile(avg_gray, sorted_tiles)

            if best_tile:
                # 6. 将对应的小图片的像素值写入到对应的小方块中
                new_img.paste(best_tile, (x, y))

    # 7. 保存最终的转换后的图片
    try:
        new_img.save(OUTPUT_IMAGE_PATH)
        print(f"Successfully created photo mosaic and saved to {OUTPUT_IMAGE_PATH}")
    except Exception as e:
        print(f"Error saving image: {e}")


# --- 执行主流程 ---
if __name__ == "__main__":
    # 1. 加载并处理所有小图
    tiles_data = load_tiles()
    # 2. 生成马赛克图片
    if tiles_data:
        create_photo_mosaic(tiles_data)