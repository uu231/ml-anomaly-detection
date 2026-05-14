"""
analyze_distribution_shift.py

目的：
- 加载 train.csv（含标签）、test_simple.csv、test_complex.csv
- 比较特征分布，量化训练集与测试集之间的偏移程度
- 推测 test_complex 可能经历的变换类型（缩放、平移、噪声等）及其概率
- 输出结果供进一步分析
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, normaltest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置路径 ====================
DATA_DIR = "../data"          # 相对于 src 目录，可根据实际情况调整
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_SIMPLE_PATH = os.path.join(DATA_DIR, "test_simple.csv")
TEST_COMPLEX_PATH = os.path.join(DATA_DIR, "test_complex.csv")
OUTPUT_FILE = "../outputs/distribution_shift_analysis.txt"   # 输出结果文件

# 特征列识别
def get_feature_cols(df):
    return [col for col in df.columns if col.startswith('f') and col[1:].isdigit()]

# ==================== 基础统计 ====================
def basic_stats(df, features):
    """返回特征的均值、标准差、中位数、偏度等统计量 DataFrame"""
    stats = []
    for col in features:
        data = df[col].dropna().values
        stats.append({
            'feature': col,
            'mean': np.mean(data),
            'std': np.std(data),
            'median': np.median(data),
            'skew': pd.Series(data).skew(),
            'kurt': pd.Series(data).kurtosis(),
            'min': np.min(data),
            'max': np.max(data),
            'q01': np.percentile(data, 1),
            'q99': np.percentile(data, 99)
        })
    return pd.DataFrame(stats)

def compare_distributions(train_df, test_df, features):
    """对每个特征进行 KS 检验，返回统计量及 p 值"""
    results = []
    for col in features:
        train_data = train_df[col].dropna().values
        test_data = test_df[col].dropna().values
        if len(train_data) == 0 or len(test_data) == 0:
            continue
        ks_stat, ks_p = ks_2samp(train_data, test_data)
        # 均值差异（相对差异）
        mean_train = np.mean(train_data)
        mean_test = np.mean(test_data)
        mean_ratio = mean_test / mean_train if mean_train != 0 else np.inf
        # 标准差差异
        std_train = np.std(train_data)
        std_test = np.std(test_data)
        std_ratio = std_test / std_train if std_train != 0 else np.inf
        results.append({
            'feature': col,
            'ks_stat': ks_stat,
            'ks_pvalue': ks_p,
            'mean_train': mean_train,
            'mean_test': mean_test,
            'mean_ratio': mean_ratio,
            'std_train': std_train,
            'std_test': std_test,
            'std_ratio': std_ratio
        })
    return pd.DataFrame(results)

# ==================== 推测变换类型 ====================
def infer_shift_type(comparison_df):
    """
    根据均值和标准差的变化模式，推测可能的全局变换类型。
    返回各种变换的概率（0~1之间的分数，总和不必为1，表示可能性强度）
    """
    # 提取均值比和标准差比
    mean_ratios = comparison_df['mean_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
    std_ratios = comparison_df['std_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(mean_ratios) == 0:
        return {}
    
    # 全局缩放：所有特征均值比和标准差比接近同一个常数（缩放因子）
    # 测量一致性：变异系数小
    mean_ratio_cv = np.std(mean_ratios) / np.mean(mean_ratios) if np.mean(mean_ratios) != 0 else np.inf
    std_ratio_cv = np.std(std_ratios) / np.mean(std_ratios) if np.mean(std_ratios) != 0 else np.inf
    
    # 偏移：均值比偏离1但标准差比接近1
    mean_offset_magnitude = np.abs(np.mean(mean_ratios) - 1.0)
    std_ratio_mean = np.mean(std_ratios)
    
    # 噪声：标准差比显著大于1，但均值比接近1
    noise_magnitude = std_ratio_mean - 1.0 if std_ratio_mean > 1 else 0
    mean_stable = mean_offset_magnitude < 0.1
    
    # 混合：两者都显著偏离
    mixed_magnitude = (mean_offset_magnitude + noise_magnitude) / 2.0
    
    # 简单启发式概率（0~1）
    prob_scale = max(0, 1 - mean_ratio_cv) if mean_ratio_cv < 0.3 else 0.0
    prob_offset = max(0, 1 - mean_offset_magnitude / 0.5) if mean_offset_magnitude > 0 else 0.0
    prob_noise = max(0, noise_magnitude / 1.0) if noise_magnitude > 0 else 0.0
    prob_mix = max(0, mixed_magnitude / 1.0)
    
    # 确保不超过1
    prob_scale = min(1.0, prob_scale)
    prob_offset = min(1.0, prob_offset)
    prob_noise = min(1.0, prob_noise)
    prob_mix = min(1.0, prob_mix)
    
    return {
        'scale (全局缩放)': prob_scale,
        'offset (全局平移)': prob_offset,
        'noise (加性噪声)': prob_noise,
        'mixed (混合变换)': prob_mix,
        '无明显偏移': 1.0 - max(prob_scale, prob_offset, prob_noise, prob_mix)
    }

# ==================== 额外：分类器区分训练/测试 ====================
def domain_classifier_accuracy(train_df, test_df, features):
    """
    训练一个分类器区分样本来自训练集还是测试集。
    如果准确率接近100%，说明两个分布差异巨大。
    返回准确率（越高表示偏移越严重）。
    """
    train_sub = train_df[features].copy()
    test_sub = test_df[features].copy()
    train_sub['domain'] = 0
    test_sub['domain'] = 1
    combined = pd.concat([train_sub, test_sub], axis=0).dropna()
    X = combined[features]
    y = combined['domain']
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    return np.mean(scores)

# ==================== 主程序 ====================
def main():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_simple = pd.read_csv(TEST_SIMPLE_PATH)
    test_complex = pd.read_csv(TEST_COMPLEX_PATH)
    
    features = get_feature_cols(train_df)
    print(f"Found {len(features)} feature columns: {features[:5]}...")
    
    # 基础统计
    print("\n--- Basic Statistics on Train ---")
    train_stats = basic_stats(train_df, features)
    print(train_stats[['feature', 'mean', 'std', 'median']].head())
    
    # 比较 simple 与 train
    print("\n--- Comparing test_simple vs train ---")
    comp_simple = compare_distributions(train_df, test_simple, features)
    print(f"Average KS statistic: {comp_simple['ks_stat'].mean():.4f}")
    print(f"Average mean ratio: {comp_simple['mean_ratio'].mean():.4f}")
    print(f"Average std ratio: {comp_simple['std_ratio'].mean():.4f}")
    
    # 比较 complex 与 train
    print("\n--- Comparing test_complex vs train ---")
    comp_complex = compare_distributions(train_df, test_complex, features)
    print(f"Average KS statistic: {comp_complex['ks_stat'].mean():.4f}")
    print(f"Average mean ratio: {comp_complex['mean_ratio'].mean():.4f}")
    print(f"Average std ratio: {comp_complex['std_ratio'].mean():.4f}")
    
    # 推测 test_complex 的变换类型
    print("\n--- Inferring shift type for test_complex ---")
    shift_probs = infer_shift_type(comp_complex)
    for k, v in shift_probs.items():
        print(f"  {k}: {v:.2f}")
    
    # 域分类器准确率
    print("\n--- Domain Classifier Accuracy (train vs test) ---")
    acc_simple = domain_classifier_accuracy(train_df, test_simple, features)
    acc_complex = domain_classifier_accuracy(train_df, test_complex, features)
    print(f"  train vs test_simple: {acc_simple:.3f} (lower = more similar)")
    print(f"  train vs test_complex: {acc_complex:.3f} (higher = more shifted)")
    
    # 额外分析：异常样本与正常样本的特征分布差异（仅 train）
    print("\n--- Anomaly vs Normal in Train (to understand pattern) ---")
    normal_df = train_df[train_df['y'] == 0]
    anomaly_df = train_df[train_df['y'] == 1]
    if len(anomaly_df) > 0:
        comp_anom = compare_distributions(normal_df, anomaly_df, features)
        print(f"Mean KS between normal and anomaly: {comp_anom['ks_stat'].mean():.4f}")
        print("This indicates how anomalous patterns look like in feature space.")
    else:
        print("No anomalies in train? Check label distribution.")
    
    # 保存完整分析结果到文件
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        f.write("=== Distribution Shift Analysis ===\n\n")
        f.write("Train vs test_simple:\n")
        f.write(comp_simple.to_string() + "\n\n")
        f.write("Train vs test_complex:\n")
        f.write(comp_complex.to_string() + "\n\n")
        f.write("Inferred shift probabilities for test_complex:\n")
        for k, v in shift_probs.items():
            f.write(f"{k}: {v:.3f}\n")
        f.write(f"\nDomain classifier accuracy (train vs test_complex): {acc_complex:.3f}\n")
        f.write(f"Domain classifier accuracy (train vs test_simple): {acc_simple:.3f}\n")
    
    print(f"\nFull results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()