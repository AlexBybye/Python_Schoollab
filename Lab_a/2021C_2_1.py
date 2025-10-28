from __future__ import annotations
import openpyxl
import numpy as np

# ---------- 文件读取 ----------
def fetch_sheet(path: str, idx: int):
    ws = openpyxl.load_workbook(path, data_only=True).worksheets[idx]
    return [[c.value for c in row] for row in ws.iter_rows()]


root = "D:/Python/revision_of_Py/Lab_a"
raw_a0 = fetch_sheet(f"{root}/a.xlsx", 0)      # 主表 sheet0
raw_a1 = fetch_sheet(f"{root}/a.xlsx", 1)      # 主表 sheet1
raw_c0 = fetch_sheet(f"{root}/c.xlsx", 0)      # 37 家 sheet0
raw_c1 = fetch_sheet(f"{root}/c.xlsx", 1)      # 37 家 sheet1
raw_d0 = fetch_sheet(f"{root}/d.xlsx", 0)      # 24 周供应 sheet0

to_float = lambda v: 0.0 if v is None else float(v) if isinstance(v, (int, float)) else 0.0

# ---------- 第一步：402 家 240 周求和 + 排序 ----------
SUPPLIER, WEEK = 402, 240
threshold = 240 * 28200 * 0.003        # 20304.0

sum_matrix = np.zeros((SUPPLIER, 2))     # col0=和，col1=原行号（1-base）
for r in range(SUPPLIER):
    ttl = sum(to_float(raw_a1[r+1][c+2]) for c in range(WEEK))
    sum_matrix[r] = ttl, r+1

# 冒泡降序
for i in range(SUPPLIER-1):
    for j in range(SUPPLIER-1-i):
        if sum_matrix[j][0] < sum_matrix[j+1][0]:
            sum_matrix[j], sum_matrix[j+1] = sum_matrix[j+1].copy(), sum_matrix[j].copy()

# 切分点
cut = next((i+1 for i, s in enumerate(sum_matrix[:, 0]) if s < threshold), SUPPLIER)
print('选取', cut, '家供应商')
for i in range(cut):
    print(int(sum_matrix[i][1]))

# ---------- 第二步：37 家 239 周差值 ----------
FIRM, WEEK2 = 37, 239
diff_cube = np.zeros((FIRM, WEEK2, 4))   # 0=平均绝对差，1=周号，2=Δ_t，3=Δ_{t+1}

for f in range(FIRM):
    for w in range(WEEK2):
        g1 = to_float(raw_c1[f+1][w+2])
        d1 = to_float(raw_c0[f+1][w+2])
        g2 = to_float(raw_c1[f+1][w+3])
        d2 = to_float(raw_c0[f+1][w+3])
        delta1, delta2 = g1 - d1, g2 - d2
        diff_cube[f, w] = (np.abs(delta1) + np.abs(delta2)) / 2, w+1, delta1, delta2

# 每行按 col0 升序（冒泡）
for f in range(FIRM):
    for i in range(WEEK2-1):
        for j in range(WEEK2-1-i):
            if diff_cube[f, j, 0] > diff_cube[f, j+1, 0]:
                diff_cube[f, j], diff_cube[f, j+1] = diff_cube[f, j+1].copy(), diff_cube[f, j].copy()

SELECT = 12
week_slot = np.full((FIRM, SELECT, 3), -2.0)   # 0=周号，1=Δ_t，2=Δ_{t+1}

for f in range(FIRM):
    k = 0
    for rank in range(WEEK2):          # 从差值最小的开始挑
        if k == SELECT:
            break
        cand = int(diff_cube[f, rank, 1])          # 当前周号
        occupied = {int(week_slot[f, j, 0]) for j in range(SELECT) if week_slot[f, j, 0] >= 0}
        if cand-1 not in occupied and cand+1 not in occupied:
            week_slot[f, k] = cand, diff_cube[f, rank, 2], diff_cube[f, rank, 3]
            k += 1

week24 = np.zeros((FIRM, 24))
for f in range(FIRM):
    idx = 0
    for s in range(SELECT):
        w = int(week_slot[f, s, 0])
        week24[f, idx] = to_float(raw_c0[f+1][w+2]); idx += 1
        week24[f, idx] = to_float(raw_c0[f+1][w+3]); idx += 1

MAT, WEEK3 = 8, 240
avg_rec = np.zeros((MAT, 2))   # 0=均值，1=原行号
for m in range(MAT):
    zeros = total = 0
    for w in range(WEEK3):
        v = to_float(raw_a0[m+1][w+1])
        (zeros, total) = (zeros+1, total) if v == 0 else (zeros, total+v)
    avg_rec[m] = 0.0 if WEEK3 == zeros else total / (WEEK3 - zeros), m+1

# 升序排序
for i in range(MAT-1):
    for j in range(MAT-1-i):
        if avg_rec[j, 0] > avg_rec[j+1, 0]:
            avg_rec[j], avg_rec[j+1] = avg_rec[j+1].copy(), avg_rec[j].copy()

for val, mid in avg_rec:
    print(val, int(mid))

# ---------- 8. 24 周总需求 ----------
total24 = np.array([sum(to_float(raw_d0[r+1][c+2]) for r in range(FIRM)) for c in range(24)])

car = [(int(t // 6000) + (1 if t % 6000 else 0)) for t in total24]
