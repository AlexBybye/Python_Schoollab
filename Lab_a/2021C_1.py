import pandas as pd
import numpy as np
import ahpy
import matplotlib.pyplot as plt

d1 = pd.read_excel('附件1 近5年402家供应商的相关数据.xlsx', sheet_name='企业的订货量（m³）')
d2 = pd.read_excel('附件1 近5年402家供应商的相关数据.xlsx', sheet_name='供应商的供货量（m³）')
d3 = pd.read_excel('附件2 近5年8家转运商的相关数据.xlsx')

# 数据预处理
d1_sub = d1.iloc[:, 2:].fillna(0)
d2_sub = d2.iloc[:, 2:].fillna(0)

# 计算供货特征
supply_times = d2_sub.apply(lambda x: sum(x != 0), axis=1)
supply_quantity_mean = d2_sub.apply(lambda x: sum(x), axis=1) / supply_times
supply_max = d2_sub.apply(lambda x: max(x), axis=1)
d12_sub = d1_sub.subtract(d2_sub) ** 2
supply_stability = d12_sub.apply(lambda x: sum(x), axis=1) / supply_times

gap_times = [None] * d2_sub.shape[0]
for i in range(d2_sub.shape[0]):
    a = d2_sub.iloc[i, :] == 0
    gap_times[i] = (a & ~np.r_[[False], a[:-1]]).sum()

gap_weeks_mean = [None] * d2_sub.shape[0]
for i in range(d2_sub.shape[0]):
    index = [0] + list(np.where(d2_sub.iloc[i, :] != 0)[0]) + [241]
    new = np.diff(index)
    gap_weeks_mean[i] = sum(new[np.where((new != 1) & (new != 0))])

supply_weeks_mean = [None] * d2_sub.shape[0]
for i in range(d2_sub.shape[0]):
    index = np.where(d2_sub.iloc[i, :] != 0)[0]
    new = np.where(np.diff(index) == 1)[0]
    supply_weeks_mean[i] = len(new) * 2 - len(np.where(np.diff(new) == 1)[0])

df = pd.DataFrame(None, columns=list(d2_sub.columns), index=list(d2_sub.index))
for i in range(d2_sub.shape[0]):
    for j in range(d2_sub.shape[1]):
        if d1_sub.iloc[i, j] == 0:
            df.iloc[i, j] = 0
        elif (d2_sub.iloc[i, j] > d1_sub.iloc[i, j] * 0.8) and (d2_sub.iloc[i, j] < d1_sub.iloc[i, j] * 1.2):
            df.iloc[i, j] = True
        else:
            df.iloc[i, j] = False
supply_proportion = df.apply(lambda x: sum(x), axis=1) / supply_times

# 数据标准化
df = pd.DataFrame({
    '供货次数': supply_times,
    '平均供货量': supply_quantity_mean,
    '单次最大供货量': supply_max,
    '供货稳定性': supply_stability,
    '间隔次数': gap_times,
    '平均间隔周数': gap_weeks_mean,
    '平均连续供货周数': supply_weeks_mean,
    '合理供货比例': supply_proportion
})

df_positive = df[['供货次数', '平均供货量', '单次最大供货量', '平均连续供货周数', '合理供货比例']]
df_positive_norm = df_positive.apply(lambda x: (x - min(x)) / (max(x) - min(x)), axis=0)

df_negative = df[['供货稳定性', '间隔次数', '平均间隔周数']]
df_negative_norm = df_negative.apply(lambda x: (max(x) - x) / (max(x) - min(x)), axis=0)

df_norm = pd.concat([df_positive_norm, df_negative_norm], axis=1, join='inner')

# 熵权法
supply_continuity = df_norm[['间隔次数', '平均间隔周数', '平均连续供货周数']]

def norm(X):
    return X / X.sum()

supply_continuity_norm = norm(supply_continuity)

k = -(1 / np.log(supply_continuity_norm.shape[0]))

def entropy(X):
    return (X * np.log(X)).sum() * k

entropy_values = entropy(supply_continuity_norm)
dod = 1 - entropy_values
w = dod / dod.sum()
weights = w.sort_values(ascending=False)

supply_continuity_weighted = supply_continuity['间隔次数'] * weights.iloc[0] + supply_continuity['平均间隔周数'] * weights.iloc[1] + supply_continuity['平均连续供货周数'] * weights.iloc[2]
df_norm.drop(['间隔次数', '平均间隔周数', '平均连续供货周数'], axis=1, inplace=True)
df_norm['供货连续性'] = supply_continuity_weighted

# 对6个一级指标进行加权
df_norm_new = norm(df_norm)
entropy_values = entropy(df_norm_new)
dod = 1 - entropy_values
w = dod / dod.sum()
weights_entropy = w.sort_values(ascending=False)

# Topsis
def norm(X):
    return X / np.sqrt((X ** 2).sum())

norm_matrix = norm(df_norm)
w_norm_matrix = norm_matrix * weights_entropy

V_plus = w_norm_matrix.apply(max)
V_minus = w_norm_matrix.apply(min)

S_plus = np.sqrt(((w_norm_matrix - V_plus) ** 2).apply(sum, axis=1))
S_minus = np.sqrt(((w_norm_matrix - V_minus) ** 2).apply(sum, axis=1))
scores = S_minus / (S_plus + S_minus)

d2['综合得分'] = scores * 100
output = d2[['供应商ID', '综合得分']]

# AHP
comparisons = {
    ('供货次数', '平均供货量'): 3, ('供货次数', '单次最大供货量'): 5, ('供货次数', '合理供货比例'): 5, ('供货次数', '供货稳定性'): 5, ('供货次数', '供货连续性'): 5,
    ('平均供货量', '单次最大供货量'): 5, ('平均供货量', '合理供货比例'): 3, ('平均供货量', '供货稳定性'): 3, ('平均供货量', '供货连续性'): 3,
    ('单次最大供货量', '合理供货比例'): 1/3, ('单次最大供货量', '供货稳定性'): 1/3, ('单次最大供货量', '供货连续性'): 1/3,
    ('合理供货比例', '供货稳定性'): 1, ('合理供货比例', '供货连续性'): 1,
    ('供货稳定性', '供货连续性'): 1
}

cal = ahpy.Compare(name='Drinks', comparisons=comparisons, precision=3, random_index='saaty')
weights_ahp = cal.target_weights
cr = cal.consistency_ratio

# 权重融合
weights_ahp = pd.DataFrame.from_dict(weights_ahp, orient='index', columns=['AHP权重'])
results = pd.concat([weights_ahp, pd.DataFrame(weights_entropy, index=weights_entropy.index, columns=['熵权法权重'])], axis=1)
results['最终权重'] = results.apply(lambda x: (x['AHP权重'] + x['熵权法权重']) / 2, axis=1)

# 最终综合得分
d2['综合得分2'] = (df_norm['供货次数'] * results.loc['供货次数', '最终权重'] +
                   df_norm['平均供货量'] * results.loc['平均供货量', '最终权重'] +
                   df_norm['单次最大供货量'] * results.loc['单次最大供货量', '最终权重'] +
                   df_norm['合理供货比例'] * results.loc['合理供货比例', '最终权重'] +
                   df_norm['供货稳定性'] * results.loc['供货稳定性', '最终权重'] +
                   df_norm['供货连续性'] * results.loc['供货连续性', '最终权重']) * 100

output = d2[['供应商ID', '综合得分2']]
output = output.sort_values('综合得分2', ascending=False)
output.iloc[0:50, :].to_csv('scores_top50.csv', index=False)

# 可视化
df = output.iloc[0:10, :]
df = df.sort_values(by='综合得分2')
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

my_range = range(1, 11)
plt.hlines(y=my_range, xmin=0, xmax=df['综合得分2'], color='skyblue')
plt.plot(df['综合得分2'], my_range, "o")

plt.yticks(my_range, df['供应商ID'])
plt.title("综合得分前10的供应商")
plt.xlabel('供应商综合得分')
plt.ylabel('供应商ID')

plt.show()