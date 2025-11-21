# -*- coding: utf-8 -*-
"""
湖泊污染动态模拟（差分方程版）
功能：计算安大略湖污染何时降至初始 10 % 并绘制长期趋势
作者：Kimi（重排版）
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# 0. 绘图中文环境
# -------------------------------------------------
plt.rcParams["font.sans-serif"] = ["SimHei"]   # 黑体
plt.rcParams["axes.unicode_minus"] = False     # 负号正常显示

# -------------------------------------------------
# 1. 基本常量
# -------------------------------------------------
LAKE_ERIE_0   = 2500      # 伊利湖初始污染量
LAKE_ONT_0    = 2500      # 安大略湖初始污染量
TARGET_RATIO  = 0.10      # 目标比例（10 %）
MAX_SIM_YEARS = 100       # 模拟总年数

# 差分方程系数
ALPHA = 0.62              # 伊利湖留存率
BETA  = 0.87              # 安大略湖留存率
GAMMA = 0.38              # 伊利湖→安大略湖转移系数
DELTA = 25                # 安大略湖每年新增固定污染

# -------------------------------------------------
# 2. 初始化轨迹容器
# -------------------------------------------------
erie_levels = [LAKE_ERIE_0]                # 伊利湖历年记录
ont_levels  = [LAKE_ONT_0]                 # 安大略湖历年记录
target_year = None                         # 首次达标年份（None 表示未达标）

# -------------------------------------------------
# 3. 逐年迭代
# -------------------------------------------------
for yr in range(1, MAX_SIM_YEARS + 1):
    last_erie = erie_levels[-1]
    last_ont  = ont_levels[-1]

    # 差分方程更新
    cur_erie = ALPHA * last_erie
    cur_ont  = BETA * last_ont + GAMMA * last_erie + DELTA

    erie_levels.append(cur_erie)
    ont_levels.append(cur_ont)

    # 首次达标判定
    if target_year is None and cur_ont <= TARGET_RATIO * LAKE_ONT_0:
        target_year = yr

# -------------------------------------------------
# 4. 终端报告
# -------------------------------------------------
print("=== 模拟结果摘要 ===")
print(f"伊利湖初始污染 : {LAKE_ERIE_0}")
print(f"安大略湖初始污染 : {LAKE_ONT_0}")
print(f"安大略湖 10 % 目标 : {TARGET_RATIO * LAKE_ONT_0:.2f}")

if target_year:
    print(f"\n安大略湖污染在第 {target_year} 年首次降至 10 % 以下。")
    print(f"  前一年污染 : {ont_levels[target_year - 1]:.2f}")
    print(f"  达标年污染 : {ont_levels[target_year]:.2f}")
else:
    print(f"\n{MAX_SIM_YEARS} 年内未降至 10 %。")

# -------------------------------------------------
# 5. 绘制长期趋势
# -------------------------------------------------
print("\n正在生成长期趋势图 …")
time_axis = np.arange(0, MAX_SIM_YEARS + 1)
equil_ont = 2500 / 13          # 理论平衡值

plt.figure(figsize=(12, 7))
plt.plot(time_axis, erie_levels,  "b:", marker="o", markersize=3,
         label=f"伊利湖（趋于 0）")
plt.plot(time_axis, ont_levels,   "r-", marker="x", markersize=3,
         label=f"安大略湖（趋于 {equil_ont:.2f}）")

# 辅助线
plt.axhline(equil_ont,          color="r", ls="--", label=f"安大略湖平衡值 {equil_ont:.2f}")
plt.axhline(TARGET_RATIO * LAKE_ONT_0, color="g", ls=":",  label=f"10 % 目标线")

# 达标高亮
if target_year:
    plt.scatter(target_year, ont_levels[target_year],
                s=100, facecolors="none", edgecolors="g", label=f"第 {target_year} 年达标")

plt.title("湖泊污染长期动态（差分方程模拟）")
plt.xlabel("年")
plt.ylabel("污染量")
plt.legend()
plt.grid(True)
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.show()
print("绘图完毕。")