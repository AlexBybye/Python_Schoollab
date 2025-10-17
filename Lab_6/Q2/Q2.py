import csv
import json
import re
from collections import Counter
import sys
import os
FILE_PATH = 'weblog.csv'

IPV4_PATTERN = re.compile(
    r'((25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.){3}(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])'
)


def extract_and_count_ips(file_path: str, ip_column_index: int = 0, delimiter: str = ',') -> dict:

    try:
        # 使用 utf-8 编码确保处理各种字符
        with open(file_path, mode='r', newline='', encoding='utf-8') as f:
            # 严格指定分隔符
            reader = csv.reader(f, delimiter=delimiter)

            # 使用生成器函数创建惰性 IP 流，以确保低内存消耗
            def clean_ip_generator():
                header_skipped = False
                for row in reader:
                    # 假定第一行为标题行，并跳过
                    if not header_skipped:
                        header_skipped = True
                        continue

                    try:
                        # 尝试从指定列索引提取 IP 地址
                        ip_candidate = row[ip_column_index].strip()

                        # 执行严格的 IP 地址验证
                        if IPV4_PATTERN.fullmatch(ip_candidate):
                            yield ip_candidate
                        # 否则，该行数据被认为是噪音或不合法，被跳过

                    except IndexError:
                        # 处理行字段数不足的情况（列偏移或数据损坏）
                        sys.stderr.write(
                            f"警告：文件 {file_path}, 第 {reader.line_num} 行: 字段不足以访问列 {ip_column_index}\n")

                    except csv.Error as e:
                        # 捕获 CSV 格式错误，并报告行号
                        sys.stderr.write(f"错误：文件 {file_path}, 第 {reader.line_num} 行解析失败: {e}\n")
                        # 即使发生错误，也继续处理下一行
                        continue

            # 使用 C-优化过的 Counter 直接消费 IP 生成器流
            ip_counts = Counter(clean_ip_generator())

            # 将 Counter 转换为原生字典，确保 JSON 序列化兼容性
            return dict(ip_counts)

    except IOError as e:
        sys.exit(f"I/O 操作失败: {e}")
    except Exception as e:
        sys.exit(f"发生未预期的错误: {e}")


# --- 主执行逻辑 ---
if __name__ == "__main__":

    result_dict = extract_and_count_ips(FILE_PATH, ip_column_index=0, delimiter=',')

    # 序列化 (S)：使用 json.dumps() 生成 JSON 字符串用于显示
    # 使用 indent=4 参数实现美观打印，增强可读性
    if result_dict:
        json_output_string = json.dumps(
            result_dict,
            indent=4,
            ensure_ascii=True,
            sort_keys=True  # 可选：按 IP 地址排序输出，便于人工比较
        )

        # 展示最终结果
        print(json_output_string)
    else:
        print(json.dumps({"status": "error", "message": "未能统计到任何有效IP地址。"}, indent=4))
