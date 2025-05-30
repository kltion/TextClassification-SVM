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

def batch_transform(feature_extractor, texts, batch_size=5000):
    """分批变换数据以减少内存使用"""
    all_features = None
    n_samples = len(texts)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch_texts = texts[start_idx:end_idx]
        
        print(f"变换批次 {i+1}/{n_batches} ({start_idx}:{end_idx})...")
        batch_features = feature_extractor.transform(batch_texts)
        
        # 如果是第一批，初始化结果矩阵
        if all_features is None:
            if isinstance(batch_features, sparse.spmatrix):
                all_features = batch_features
            else:
                all_features = batch_features.copy()
        else:
            # 合并特征
            if isinstance(batch_features, sparse.spmatrix):
                all_features = sparse.vstack([all_features, batch_features])
            else:
                all_features = np.vstack([all_features, batch_features])
        
        # 强制垃圾回收
        del batch_texts
        if i % 2 == 0:  # 更频繁地收集垃圾
            gc.collect()
            memory_stats()
        
    return all_features

def batch_predict(model, texts, batch_size=3000):
    """分批预测以减少内存使用"""
    all_predictions = []
    n_samples = len(texts)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch_texts = texts[start_idx:end_idx]
        
        print(f"预测批次 {i+1}/{n_batches} ({start_idx}:{end_idx})...")
        batch_predictions = model.predict(batch_texts)
        all_predictions.extend(batch_predictions)
        
        # 强制垃圾回收
        del batch_texts
        gc.collect()
        if i % 5 == 0:
            memory_stats()
        
    return np.array(all_predictions)

def incremental_train_svm(X_train_transformed, y_train, classifier, batch_size=5000):
    """增量训练SVM模型以减少内存使用"""
    n_samples = X_train_transformed.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # 确保稀疏矩阵格式一致
    if not isinstance(X_train_transformed, sparse.csr_matrix):
        if isinstance(X_train_transformed, sparse.spmatrix):
            X_train_transformed = X_train_transformed.tocsr()
    
    # 首先使用小批量数据进行初始拟合
    first_batch_size = min(batch_size, n_samples)
    print(f"初始拟合使用前 {first_batch_size} 个样本...")
    classifier.fit(
        X_train_transformed[:first_batch_size], 
        y_train[:first_batch_size]
    )
    
    # 然后使用其余数据进行部分拟合
    if n_samples > first_batch_size:
        for i in range(1, n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            print(f"增量拟合批次 {i+1}/{n_batches} ({start_idx}:{end_idx})...")
            
            batch_X = X_train_transformed[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]
            
            # 使用warm_start重新拟合，内部会使用之前的系数作为初始值
            # 对于LinearSVC，我们需要手动实现增量训练
            prev_coef = classifier.coef_.copy()
            prev_intercept = classifier.intercept_.copy()
            
            # 仅使用当前批次拟合模型
            classifier.fit(batch_X, batch_y)
            
            # 平均新旧模型的系数和截距
            # 这是一种简单的增量更新方法
            weight = (end_idx - start_idx) / end_idx  # 当前批次的权重
            classifier.coef_ = (1 - weight) * prev_coef + weight * classifier.coef_
            classifier.intercept_ = (1 - weight) * prev_intercept + weight * classifier.intercept_
            
            # 释放内存
            del batch_X, batch_y, prev_coef, prev_intercept
            gc.collect()
            memory_stats()
    
    return classifier

def train_svm_model(X_train, y_train, X_val, y_val):
    """训练SVM模型"""
    print("开始训练SVM模型...")
    memory_stats()
    start_time = time.time()
    
    # 构建特征提取流水线
    features = build_feature_pipeline()
    
    # 构建分类器
    classifier = LinearSVC(
        C=1.0, 
        class_weight='balanced', 
        dual=False,
        max_iter=2000,       # 减少迭代次数以加快训练
        tol=1e-3,            # 增加容忍度以更快收敛
        loss='squared_hinge',
        random_state=42
    )
    
    # 构建完整流水线
    pipeline = Pipeline([
        ('features', features),
        ('classifier', classifier)
    ])
    
    # 拟合特征提取器
    print("拟合特征提取器...")
    # 使用小批量数据拟合特征提取器
    sample_size = min(30000, len(X_train))
    sample_indices = np.random.choice(range(len(X_train)), sample_size, replace=False)
    X_train_sample = [X_train.iloc[i] for i in sample_indices]
    y_train_sample = [y_train.iloc[i] for i in sample_indices]
    
    # 拟合特征提取器
    with parallel_backend('threading', n_jobs=N_JOBS):
        pipeline.named_steps['features'].fit(X_train_sample, y_train_sample)
    
    # 释放样本内存
    del X_train_sample, y_train_sample
    gc.collect()
    memory_stats()
    
    # 分批变换训练数据
    print("变换训练数据...")
    X_train_transformed = batch_transform(
        pipeline.named_steps['features'], 
        X_train, 
        batch_size=3000  # 更小的批处理大小
    )
    
    # 执行增量训练
    print("增量训练分类器...")
    y_train_array = np.array(y_train)
    
    with parallel_backend('threading', n_jobs=N_JOBS):
        incremental_train_svm(
            X_train_transformed, 
            y_train_array, 
            pipeline.named_steps['classifier'], 
            batch_size=5000
        )
    
    # 释放训练数据内存
    del X_train_transformed, y_train_array
    gc.collect()
    memory_stats()
    
    # 评估模型
    print("评估模型...")
    y_pred = batch_predict(pipeline, X_val, batch_size=3000)
    val_f1 = f1_score(y_val, y_pred, average='macro')
    print(f"验证集F1分数: {val_f1:.4f}")
    print(classification_report(y_val, y_pred))
    
    # 保存混淆矩阵图
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f"{i}-{LABEL_MAPPING[i]}" for i in sorted(set(y_val))], 
                yticklabels=[f"{i}-{LABEL_MAPPING[i]}" for i in sorted(set(y_val))])
    plt.title('混淆矩阵')
    plt.ylabel('实际标签')
    plt.xlabel('预测标签')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/confusion_matrix.png')
    plt.close()
    
    # 保存模型
    print("保存模型...")
    joblib.dump(pipeline, f'{OUTPUT_DIR}/svm_model.pkl')
    
    # 清理内存
    del y_pred
    gc.collect()
    memory_stats()
    
    print(f"模型训练完成，耗时 {time.time() - start_time:.2f} 秒")
    
    return pipeline

def predict_and_submit(model, test_texts):
    """使用模型预测测试集并生成提交文件"""
    # 预测测试集
    print("预测测试集...")
    memory_stats()
    
    # 使用批处理预测
    predictions = batch_predict(model, test_texts, batch_size=2000)
    
    # 创建提交文件
    submission = pd.DataFrame({'label': predictions})
    
    # 查看标签分布
    print("预测标签分布:")
    pred_counts = submission['label'].value_counts().sort_index()
    print(pred_counts)
    
    # 生成提交文件（带标题）
    submission_path = f'{OUTPUT_DIR}/submission.csv'
    submission.to_csv(submission_path, index=False)
    
    # 生成无标题版本的提交文件
    submission_noheader_path = f'{OUTPUT_DIR}/submission_noheader.csv'
    submission.to_csv(submission_noheader_path, index=False, header=False)
    
    print(f"提交文件已生成:")
    print(f"- 带标题版本: {submission_path}")
    print(f"- 无标题版本: {submission_noheader_path}")
    
    # 计算标签分布与训练集的差异，检查是否有异常
    print("检查标签分布是否合理...")
    # 分批读取训练集以减少内存使用
    label_counts = Counter()
    chunks = pd.read_csv(f'{INPUT_DIR}/train_set.csv', sep='\t', usecols=['label'], chunksize=10000)
    for chunk in chunks:
        label_counts.update(chunk['label'])
    total = sum(label_counts.values())
    
    train_dist = {label: count/total for label, count in label_counts.items()}
    pred_dist = submission['label'].value_counts(normalize=True).to_dict()
    
    print("训练集和预测集标签分布比较:")
    for label in sorted(train_dist.keys()):
        train_pct = train_dist.get(label, 0) * 100
        pred_pct = pred_dist.get(label, 0) * 100
        diff = pred_pct - train_pct
        print(f"标签 {label} ({LABEL_MAPPING.get(label, '未知')}): 训练集 {train_pct:.2f}%, 预测集 {pred_pct:.2f}%, 差异 {diff:.2f}%")

def main():
    print("开始执行...")
    print(f"使用 {N_JOBS} 个CPU核心进行并行处理")
    
    # 初始化内存监控
    memory_stats()
    
    # 控制NumPy线程 - 通过环境变量设置而不是直接调用set_num_threads
    # NumPy已经会读取OMP_NUM_THREADS环境变量
    # 所以我们不需要显式调用set_num_threads
    
    # 加载数据
    train_df, test_df = load_data()
    
    if train_df is not None:
        print("数据预处理...")
        X_train, X_val, y_train, y_val, test_texts = preprocess_data(train_df, test_df)
        
        # 释放原始数据，节省内存
        del train_df
        gc.collect()
        memory_stats()
        
        if X_train is not None:
            print("训练SVM模型...")
            model = train_svm_model(X_train, y_train, X_val, y_val)
            
            # 释放训练和验证数据，节省内存
            del X_train, X_val, y_train, y_val
            gc.collect()
            memory_stats()
            
            print("生成提交文件...")
            predict_and_submit(model, test_texts)
            
            print("全部完成!")
        else:
            print("由于数据预处理出错，无法继续执行。")
    else:
        print("由于数据加载出错，无法继续执行。")

if __name__ == "__main__":
    main()