import pandas as pd

# 读取 CSV 文件
file_path = '附件2.csv'  # 替换为您的文件路径
data = pd.read_csv(file_path, encoding='gbk')

# 提取交叉口为 "经中路-纬中路" 的所有行
filtered_data = data[data['交叉口'] == '经中路-纬中路']

# 将结果保存为新的 CSV 文件
output_file_path = 'filtered_1.csv'
filtered_data.to_csv(output_file_path, index=False)

print(f"过滤后的数据已保存至 {output_file_path}")
