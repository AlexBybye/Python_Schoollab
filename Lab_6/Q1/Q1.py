import pandas as pd

file_path = 'data.csv'

df = pd.read_csv(file_path)

# 使用正则表达式清洗人均价格
df['人均价格'] = df['人均价格'].astype(str).str.replace(r'[^\d.]', '', regex=True)

# 转换为浮点数，处理转换过程中可能出现的错误
df['人均价格'] = pd.to_numeric(df['人均价格'], errors='coerce')

# 删除清洗后价格仍为 NaN 的行
df.dropna(subset=['人均价格'], inplace=True)
# 餐厅种类人均价格平均值
type_avg_price = df.groupby('餐厅种类')['人均价格'].mean().sort_values(ascending=False)

# 所在地区人均价格平均值
region_avg_price = df.groupby('所在地区')['人均价格'].mean().sort_values(ascending=False)

# 评分人均价格平均值
rating_avg_price = df.groupby('评分')['人均价格'].mean().sort_values(ascending=False)

print("\n--- 1. 餐厅种类平均人均价格 ---\n", type_avg_price)
print("\n--- 2. 所在地区平均人均价格 ---\n", region_avg_price)
print("\n--- 3. 评分平均人均价格 ---\n", rating_avg_price)