#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
import joblib
import time
import matplotlib
matplotlib.use('Agg')  # 避免使用GUI后端
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import gc  # 添加垃圾回收
import warnings
from joblib import parallel_backend, Parallel, delayed
import psutil
import multiprocessing
from scipy import sparse
warnings.filterwarnings('ignore')

# 设置并行参数
N_JOBS = max(1, multiprocessing.cpu_count() - 1)  # 保留一个核心给系统使用
BATCH_SIZE = 10000  # 减小批处理大小以减少内存使用
MAX_FEATURES_TFIDF = 25000  # 减少特征数量
MAX_FEATURES_COUNT = 10000  # 减少特征数量

# 路径设置
INPUT_DIR = './input'
OUTPUT_DIR = './output'

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 标签映射
LABEL_MAPPING = {
    0: '科技', 1: '股票', 2: '体育', 3: '娱乐', 4: '时政', 
    5: '社会', 6: '教育', 7: '财经', 8: '家居', 9: '游戏', 
    10: '房产', 11: '时尚', 12: '彩票', 13: '星座'
}

def memory_stats():
    """显示当前内存使用情况"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"内存使用: {mem_info.rss / (1024 * 1024):.2f} MB")
    
    vm = psutil.virtual_memory()
    print(f"系统内存: 总计={vm.total/(1024*1024*1024):.1f}GB, 已用={vm.percent}%, 可用={vm.available/(1024*1024*1024):.1f}GB")

def load_data():
    """加载训练集和测试集数据"""
    try:
        print("开始加载数据...")
        memory_stats()
        
        # 分批加载训练数据以减少内存压力
        print("加载训练集...")
        train_chunks = pd.read_csv(f'{INPUT_DIR}/train_set.csv', sep='\t', chunksize=BATCH_SIZE)
        train_df_list = []
        for i, chunk in enumerate(train_chunks):
            train_df_list.append(chunk)
            if (i+1) % 5 == 0:
                print(f"已加载 {(i+1)*BATCH_SIZE} 条训练数据")
                memory_stats()
                # 如果内存使用超过70%，停止加载更多数据
                if psutil.virtual_memory().percent > 70:
                    print("内存使用率过高，停止加载更多数据")
                    break
        train_df = pd.concat(train_df_list, ignore_index=True)
        del train_df_list
        gc.collect()
        
        print("加载测试集...")
        test_df = pd.read_csv(f'{INPUT_DIR}/test_a.csv', sep='\t')
        
        print(f"训练集大小: {train_df.shape}, 测试集大小: {test_df.shape}")
        memory_stats()
        return train_df, test_df
    except Exception as e:
        print(f"加载数据出错: {e}")
        return None, None

def analyze_data(train_df, sample_size=5000):
    """分析训练数据的特征 (使用样本避免内存溢出)"""
    print("\n数据分析:")
    memory_stats()
    
    # 检查标签分布
    print("标签分布:")
    label_counts = train_df['label'].value_counts().sort_index()
    print(label_counts)
    print("\n标签含义:")
    for label, meaning in LABEL_MAPPING.items():
        print(f"{label}: {meaning}")
    
    # 分析文本长度 (只使用样本)
    sample_df = train_df.sample(min(sample_size, len(train_df)), random_state=42)
    sample_df['text_len'] = sample_df['text'].apply(lambda x: len(str(x).split()))
    
    print("\n文本长度统计 (基于样本):")
    print(f"平均长度: {sample_df['text_len'].mean():.2f}")
    print(f"最大长度: {sample_df['text_len'].max()}")
    print(f"最小长度: {sample_df['text_len'].min()}")
    
    # 分析词频 (只使用样本的前500条)
    mini_sample = sample_df.head(500)
    all_words = []
    for text in mini_sample['text']:
        all_words.extend(str(text).split())
    word_counts = Counter(all_words)
    print(f"\n不同词的数量 (基于小样本): {len(word_counts)}")
    print("最常见的10个词:")
    print(word_counts.most_common(10))
    
    # 可视化标签分布
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(x='label', data=train_df)
    plt.title('标签分布')
    plt.xlabel('标签')
    plt.ylabel('数量')
    
    # 添加标签名称
    for i, v in enumerate(label_counts):
        ax.text(i, v + 5, LABEL_MAPPING.get(i, '未知'), ha='center', rotation=90)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/label_distribution.png')
    plt.close()
    
    # 可视化文本长度分布
    plt.figure(figsize=(12, 6))
    sns.histplot(sample_df['text_len'], bins=50)
    plt.title('文本长度分布 (基于样本)')
    plt.savefig(f'{OUTPUT_DIR}/text_length_distribution.png')
    plt.close()
    
    # 释放内存
    del sample_df, mini_sample, all_words, word_counts
    gc.collect()
    memory_stats()
    
    return label_counts

class TextStatisticsExtractor:
    """提取文本的统计特征"""
    def __init__(self, n_jobs=N_JOBS):
        self.n_jobs = n_jobs
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, texts):
        # 使用并行处理
        def process_text(text):
            words = str(text).split()
            return {
                'length': len(words),
                'unique_ratio': len(set(words)) / (len(words) + 1)  # 避免除零
            }
            
        # 分批处理以减少内存压力
        features = []
        batch_size = 5000  # 更小的批处理大小
        n_samples = len(texts)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch_texts = texts[start_idx:end_idx]
            with parallel_backend('threading', n_jobs=self.n_jobs):
                batch_features = Parallel()(
                    delayed(process_text)(text) for text in batch_texts
                )
            features.extend(batch_features)
            
            # 手动垃圾回收
            if i % 2 == 0:  # 更频繁地收集垃圾
                gc.collect()
                
        return pd.DataFrame(features)

def preprocess_data(train_df, test_df):
    """数据预处理"""
    if train_df is None or test_df is None:
        return None, None, None, None, None
    
    print("开始数据预处理...")
    memory_stats()
    
    # 分析数据
    analyze_data(train_df)
    
    # 如果数据集太大，可以考虑使用分层抽样减少数据量
    MAX_TRAIN_SAMPLES = 100000  # 更加严格地限制训练样本数量
    if len(train_df) > MAX_TRAIN_SAMPLES:
        print(f"训练集较大，进行分层抽样减少到约 {MAX_TRAIN_SAMPLES} 条...")
        train_df_sampled = train_df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(MAX_TRAIN_SAMPLES * len(x) / len(train_df))), random_state=42)
        )
        print(f"抽样后训练集大小: {train_df_sampled.shape}")
        X_train_full = train_df_sampled['text']
        y_train_full = train_df_sampled['label']
    else:
        X_train_full = train_df['text']
        y_train_full = train_df['label']
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.1, random_state=42, stratify=y_train_full
    )
    
    print(f"训练集大小: {X_train.shape}, 验证集大小: {X_val.shape}")
    memory_stats()
    
    # 释放内存
    if 'train_df_sampled' in locals():
        del train_df_sampled
    del X_train_full, y_train_full
    gc.collect()
    
    return X_train, X_val, y_train, y_val, test_df['text']

def build_feature_pipeline():
    """构建特征提取流水线"""
    # TF-IDF特征 - 进一步减少特征数量以节省内存
    tfidf = TfidfVectorizer(
        ngram_range=(1, 1),  # 仅使用单个词，不使用二元组合
        max_features=MAX_FEATURES_TFIDF,  # 减少特征数量
        min_df=10,           # 增加最小文档频率以过滤罕见词
        max_df=0.9,
        use_idf=True,
        sublinear_tf=True,
        dtype=np.float32     # 使用float32而不是float64以节省内存
    )
    
    # 特征联合 - 移除Count Vectorizer以减少特征数量
    features = Pipeline([
        ('tfidf', tfidf)
    ])
    
    return features