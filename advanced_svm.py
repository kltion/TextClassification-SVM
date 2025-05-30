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