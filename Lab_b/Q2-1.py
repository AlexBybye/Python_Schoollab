import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import f_regression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def data_preprocessing(file_path):
    df = pd.read_excel(file_path)

    y_column = None
    for col in df.columns:
        if 'Y染色体浓度' in col or 'y染色体浓度' in col.lower():
            y_column = col
            break

    if y_column is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        y_column = numeric_cols[-1]

    try:
        df_male = df[df[y_column] > 0].copy()
    except:
        df_male = df.copy()

    def convert_gestational_week(week_str):
        if isinstance(week_str, str):
            week_str_lower = week_str.lower()
            if 'w' in week_str_lower:
                w_pos = week_str_lower.find('w')
                try:
                    weeks = int(week_str[:w_pos])
                    if len(week_str) > w_pos + 1 and '+' in week_str[w_pos+1:]:
                        plus_pos = week_str[w_pos+1:].find('+') + w_pos + 1
                        try:
                            days = int(week_str[plus_pos+1:])
                            return weeks + days / 7
                        except:
                            return weeks
                    return weeks
                except:
                    return np.nan
        try:
            return float(week_str) if pd.notna(week_str) else np.nan
        except (ValueError, TypeError):
            return np.nan

    week_column = None
    for col in df.columns:
        if '检测孕周' in col or '孕周' in col:
            week_column = col
            break

    if week_column:
        df_male['孕周数值'] = df_male[week_column].apply(convert_gestational_week)
    else:
        df_male['孕周数值'] = 15

    df_filtered = df_male[(df_male['孕周数值'] >= 10) & (df_male['孕周数值'] <= 25)].copy()

    try:
        y_min, y_max = df_filtered[y_column].quantile([0.01, 0.99])
        df_filtered = df_filtered[(df_filtered[y_column] >= y_min) & (df_filtered[y_column] <= y_max)]
    except:
        pass

    # 选择关键变量
    selected_vars = ['孕周数值']

    for col in df.columns:
        if 'BMI' in col or 'bmi' in col.lower():
            selected_vars.append(col)
            break

    for col in df.columns:
        if '年龄' in col:
            selected_vars.append(col)
            break

    selected_vars.append(y_column)

    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in selected_vars and col != y_column and 'id' not in col.lower() and '序号' not in col:
            if len(selected_vars) < 8:
                selected_vars.append(col)

    df_model = df_filtered[selected_vars].copy()

    rename_dict = {y_column: 'Y染色体浓度'}
    for col in df_model.columns:
        if 'BMI' in col or 'bmi' in col.lower():
            rename_dict[col] = '孕妇BMI'
        elif '年龄' in col:
            rename_dict[col] = '年龄'

    df_model = df_model.rename(columns=rename_dict)

    df_model = df_model.dropna()

    return df_model

def correlation_analysis(df_model):
    y_concentration = df_model['Y染色体浓度']
    shapiro_stat, shapiro_p = stats.shapiro(y_concentration)

    # 根据正态性选择相关系数方法
    corr_method = 'spearman' if shapiro_p < 0.05 else 'pearson'

    # 计算相关系数矩阵
    corr_matrix = df_model.corr(method=corr_method)

    y_corr = corr_matrix['Y染色体浓度'].drop('Y染色体浓度', errors='ignore')

    df_model['BMI分组'] = pd.cut(df_model['孕妇BMI'],
                              bins=[0, 28, 32, 36, 40, np.inf],
                              labels=['BMI<28', '28≤BMI<32', '32≤BMI<36', '36≤BMI<40', 'BMI≥40'])

    groups = [group['Y染色体浓度'].values for name, group in df_model.groupby('BMI分组')]
    f_stat, f_p = stats.f_oneway(*groups)

    # 检查多重共线性
    high_corr_vars = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (abs(corr_matrix.iloc[i, j]) > 0.7 and
                corr_matrix.columns[i] != 'Y染色体浓度' and
                corr_matrix.columns[j] != 'Y染色体浓度'):
                high_corr_vars.append((corr_matrix.columns[i], corr_matrix.columns[j]))

    return df_model, y_corr, high_corr_vars, y_corr.loc[['孕周数值', '孕妇BMI']].to_dict() if '孕妇BMI' in y_corr.index else None, f_stat, f_p

def build_models(df_model, y_corr, high_corr_vars):
    X = df_model.drop(['Y染色体浓度', 'BMI分组'], axis=1)
    y = df_model['Y染色体浓度']

    critical_vars = ['孕周数值', '孕妇BMI']

    # 处理多重共线性，但保留关键变量
    to_drop = set()
    for var1, var2 in high_corr_vars:
        if var1 in critical_vars and var2 in critical_vars:
            if abs(y_corr.get(var1, 0)) < abs(y_corr.get(var2, 0)):
                to_drop.add(var1)
            else:
                to_drop.add(var2)
        elif var1 in critical_vars:
            to_drop.add(var2)
        elif var2 in critical_vars:
            to_drop.add(var1)
        else:
            if abs(y_corr.get(var1, 0)) < abs(y_corr.get(var2, 0)):
                to_drop.add(var1)
            else:
                to_drop.add(var2)

    if to_drop:
        X = X.drop(to_drop, axis=1)

    linear_model = LinearRegression()
    linear_model.fit(X, y)

    y_pred_linear = linear_model.predict(X)
    r2_linear = r2_score(y, y_pred_linear)
    n = len(X)
    k = len(X.columns)
    adjusted_r2_linear = 1 - (1 - r2_linear) * (n - 1) / (n - k - 1)

    # 多项式回归
    best_model = linear_model
    best_X = X
    best_feature_names = X.columns
    best_model_type = 'linear'
    best_r2 = adjusted_r2_linear

    try:
        # 使用关键变量：孕周数和BMI
        critical_vars = ['孕周数值', '孕妇BMI']
        continuous_vars = [var for var in critical_vars if var in X.columns]

        if len(continuous_vars) < 2:
            for col in X.columns:
                if (col not in continuous_vars and
                    pd.api.types.is_numeric_dtype(X[col]) and
                    X[col].std() > 0):
                    continuous_vars.append(col)
                    if len(continuous_vars) >= 2:
                        break

        if continuous_vars:
            X_core = X[continuous_vars].copy()

            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X_core)
            feature_names = poly.get_feature_names_out(continuous_vars)

            other_features = X.drop(continuous_vars, axis=1, errors='ignore')
            if not other_features.empty:
                X_poly = np.hstack([X_poly, other_features.values])
                feature_names = np.append(feature_names, other_features.columns)

            # 构建多项式模型
            poly_model = LinearRegression()
            poly_model.fit(X_poly, y)

            # 评估多项式模型
            y_pred_poly = poly_model.predict(X_poly)
            r2_poly = r2_score(y, y_pred_poly)
            k_poly = X_poly.shape[1]
            adjusted_r2_poly = 1 - (1 - r2_poly) * (n - 1) / (max(1, n - k_poly - 1))

            # 返回最佳模型
            if adjusted_r2_poly > adjusted_r2_linear:
                best_model = poly_model
                best_X = X_poly
                best_feature_names = feature_names
                best_model_type = 'poly'
                best_r2 = adjusted_r2_poly
    except Exception:
        pass

    return best_model, best_X, y, best_feature_names, best_model_type, best_r2, r2_linear, adjusted_r2_linear

def significance_testing(model, X, y, feature_names, model_type, best_r2, r2_linear, adjusted_r2_linear, key_corr, f_stat_bmi, f_p_bmi):
    # 计算模型的整体F统计量
    n = len(y)
    k = X.shape[1]
    y_pred = model.predict(X)
    rss = np.sum((y - y_pred) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)
    f_stat = ((tss - rss) / k) / (rss / (n - k - 1))

    # 计算p值
    from scipy.stats import f
    p_value = 1 - f.cdf(f_stat, k, n - k - 1)

    # 残差正态性检验
    residuals = y - y_pred
    shapiro_stat, shapiro_p = stats.shapiro(residuals)

    bmi_coef = None
    gestational_coef = None
    has_gestational_squared = False
    has_bmi_squared = False

    for i, feature in enumerate(feature_names):
        if 'BMI' in feature and not ('^' in feature or ' ' in feature):
            bmi_coef = model.coef_[i]
        elif '周' in feature and not ('^' in feature or ' ' in feature):
            gestational_coef = model.coef_[i]
        elif '孕周数值^2' in feature:
            has_gestational_squared = True
        elif '孕妇BMI^2' in feature:
            has_bmi_squared = True

    print("NIPT数据分析：胎儿Y染色体浓度与孕妇指标关系研究")

    if key_corr:
        print("\n与Y染色体浓度的相关系数:")
        for var, corr in key_corr.items():
            print(f"{var}: {corr:.4f} ")

    print(f"\nBMI分组的方差分析: F统计量={f_stat_bmi:.4f}, p值={f_p_bmi:.4f}")

    print(f"\n最终选择的模型类型: {'多项式回归' if model_type == 'poly' else '线性回归'}")
    print(f"线性回归R方: {r2_linear:.4f}, 调整后R方: {adjusted_r2_linear:.4f}")
    print(f"最佳模型调整后R方: {best_r2:.4f}")
    print(f"模型整体显著性检验（F检验）: F={f_stat:.4f}, p={p_value:.6f}")
    print(f"残差正态性检验: p值={shapiro_p:.4f}")

    print("\n关键变量分析：")
    if gestational_coef is not None:
        gest_direction = "正相关" if gestational_coef > 0 else "负相关"
        strength = "强" if abs(gestational_coef) > 0.01 else "中等" if abs(gestational_coef) < 0.01 and abs(gestational_coef) > 0.001 else "弱"
        print(f"1. 孕周数对Y染色体浓度的影响:")
        print(f"   - 相关方向: {gest_direction}")
        print(f"   - 回归系数: {gestational_coef:.6f}")
        print(f"   - 相关性强度: {strength}")
        if has_gestational_squared:
            print(f"   - 存在非线性关系，表明孕周数对Y染色体浓度的影响不是简单线性的")

    if bmi_coef is not None:
        bmi_direction = "正相关" if bmi_coef > 0 else "负相关"
        strength = "强" if abs(bmi_coef) > 0.01 else "中等" if abs(bmi_coef) < 0.01 and abs(bmi_coef) > 0.001 else "弱"
        print(f"\n2. BMI对Y染色体浓度的影响:")
        print(f"   - 相关方向: {bmi_direction}")
        print(f"   - 回归系数: {bmi_coef:.6f}")
        print(f"   - 相关性强度: {strength}")
        if has_bmi_squared:
            print(f"   - 存在非线性关系，表明BMI对Y染色体浓度的影响呈现曲线关系")

# ================= 可视化增强（追加到原文件末尾即可） =================
import matplotlib.pyplot as plt
import seaborn as sns
import os
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid", {'axes.edgecolor': '.2'})


class NIPTAnalyzer:
    """
    把原脚本所有函数封装成类，新增 plot 方法
    """
    def __init__(self, file_path='附件.xlsx'):
        # 此时 correlation_analysis / data_preprocessing 已定义
        self.df_model, self.y_corr, self.high_corr_vars, self.key_corr, self.f_stat_bmi, self.f_p_bmi = \
            correlation_analysis(data_preprocessing(file_path))
        self.model, self.X, self.y, self.feature_names, self.model_type, self.best_r2, self.r2_linear, self.adj_r2_linear = \
            build_models(self.df_model, self.y_corr, self.high_corr_vars)
        self.y_pred = self.model.predict(self.X)
        self.residuals = self.y - self.y_pred

    def print_result(self):
        significance_testing(self.model, self.X, self.y, self.feature_names,
                             self.model_type, self.best_r2, self.r2_linear,
                             self.adj_r2_linear, self.key_corr,
                             self.f_stat_bmi, self.f_p_bmi)

    def plot_all(self, save_dir='NIPT_plots'):
        os.makedirs(save_dir, exist_ok=True)

        # 1. Y 浓度分布
        plt.figure(figsize=(5, 4))
        sns.histplot(self.y, bins=30, kde=True, color='steelblue')
        plt.title('Y 染色体浓度分布')
        plt.savefig(os.path.join(save_dir, '1_Y_dist.png'), dpi=300)
        plt.close()

        # 2. 孕周分布
        plt.figure(figsize=(5, 4))
        sns.histplot(self.df_model['孕周数值'], bins=25, kde=True, color='seagreen')
        plt.title('孕周数值分布')
        plt.savefig(os.path.join(save_dir, '2_GA_dist.png'), dpi=300)
        plt.close()

        # 3. BMI 分布
        if '孕妇BMI' in self.df_model:
            plt.figure(figsize=(5, 4))
            sns.histplot(self.df_model['孕妇BMI'], bins=25, kde=True, color='coral')
            plt.title('孕妇 BMI 分布')
            plt.savefig(os.path.join(save_dir, '3_BMI_dist.png'), dpi=300)
            plt.close()

        # 4. 相关矩阵
        plt.figure(figsize=(6, 5))
        corr_df = self.df_model.drop(columns=['BMI分组'], errors='ignore').corr(method='spearman')
        mask = np.triu(np.ones_like(corr_df, dtype=bool))
        sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('变量相关矩阵 (Spearman)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '4_corr_matrix.png'), dpi=300)
        plt.close()

        # 5. 实测 vs 预测
        plt.figure(figsize=(5, 5))
        sns.scatterplot(x=self.y_pred, y=self.y, alpha=0.6, color='darkcyan')
        plt.plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], 'r--')
        plt.xlabel('预测 Y 浓度')
        plt.ylabel('实测 Y 浓度')
        plt.title(f'实测 vs 预测（{"多项式" if self.model_type=="poly" else "线性"}）')
        plt.text(0.05, 0.95, f'$R^2_{{adj}}={self.best_r2:.3f}$', transform=plt.gca().transAxes,
                 bbox=dict(boxstyle='round', facecolor='w'))
        plt.savefig(os.path.join(save_dir, '5_pred_vs_actual.png'), dpi=300)
        plt.close()

        # 6. 残差 QQ
        plt.figure(figsize=(5, 4))
        stats.probplot(self.residuals, dist="norm", plot=plt)
        plt.title('残差正态 QQ-Plot')
        plt.savefig(os.path.join(save_dir, '6_residual_QQ.png'), dpi=300)
        plt.close()

        print(f'>>> 全部 6 张图已保存至文件夹：{os.path.abspath(save_dir)}')


# 新的主入口，覆盖掉原 main()
def main():
    try:
        analyzer = NIPTAnalyzer('附件.xlsx')
        analyzer.print_result()   # 原控制台输出
        analyzer.plot_all()       # 新增可视化
    except Exception as e:
        print('分析过程出错：', e)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()