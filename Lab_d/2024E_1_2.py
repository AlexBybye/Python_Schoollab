import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取筛选后的 CSV 文件
data = pd.read_csv('filtered_1.csv')

# 将时间列转换为 datetime 格式
data['时间'] = pd.to_datetime(data['时间'])

# 创建小时列，提取时间中的小时信息
data['小时'] = data['时间'].dt.hour

# 按小时统计车流量
flow_per_hour = data.groupby('小时').size().reset_index(name='车流量')

# 数据归一化处理，确保聚类不受量纲影响
scaler = StandardScaler()
flow_scaled = scaler.fit_transform(flow_per_hour[['车流量']])

# 寻找最佳的聚类数量：使用肘部法则
inertia_list = []
silhouette_scores = []
cluster_range = range(2, 10)

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(flow_scaled)
    inertia_list.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(flow_scaled, kmeans.labels_))

# 使用肘部法则绘制图表
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, inertia_list, marker='o', label='肘部法则')
plt.title('肘部法则选择最佳聚类数量')
plt.xlabel('聚类数量')
plt.ylabel('Inertia（惯性）')
plt.grid(True)
plt.show()

# 使用轮廓系数绘制图表
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, silhouette_scores, marker='o', label='轮廓系数')
plt.title('轮廓系数选择最佳聚类数量')
plt.xlabel('聚类数量')
plt.ylabel('轮廓系数')
plt.grid(True)
plt.show()

optimal_clusters = 2
kmeans_optimal = KMeans(n_clusters=optimal_clusters)
flow_per_hour['聚类标签'] = kmeans_optimal.fit_predict(flow_scaled)

plt.figure(figsize=(12, 6))

# 用聚类标签当颜色映射
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=flow_per_hour['聚类标签'].min(),
                     vmax=flow_per_hour['聚类标签'].max())
colors = cmap(norm(flow_per_hour['聚类标签']))

bars = plt.bar(flow_per_hour['小时'], flow_per_hour['车流量'],
               color=colors, edgecolor='grey', linewidth=0.5)

# 加颜色条当图例
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])          # 仅用于显示色条
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('聚类标签')

plt.title('分小时车流量（按聚类着色）')
plt.xlabel('小时')
plt.ylabel('车流量')
plt.xticks(range(0, 24))
plt.grid(axis='y', ls='--', alpha=0.6)
plt.tight_layout()
plt.show()
# 查看聚类结果
print(flow_per_hour)

