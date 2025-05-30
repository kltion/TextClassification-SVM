#!/bin/bash

# 设置CPU性能模式
echo "设置CPU性能参数..."
if [ -w /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor ]; then
    echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null 2>&1
    echo "CPU性能模式已设置"
else
    echo "无法设置CPU性能模式，继续执行..."
fi

# 清理系统缓存以释放内存
echo "清理系统缓存..."
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' || echo "无法清理缓存，继续执行..."

# 调整进程优先级
echo "设置进程优先级..."
export PYTHONIOENCODING=utf-8
# 设置NumPy和其他库的线程数
export OMP_NUM_THREADS=$(nproc --ignore=1)  # 使用所有核心减一
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export VECLIB_MAXIMUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

# 安装必要的依赖
echo "安装依赖..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    pip install -q joblib scikit-learn pandas matplotlib seaborn psutil || echo "依赖安装失败，继续执行..."
fi

# 显示系统资源信息
echo "系统资源信息:"
echo "CPU核心数: $(nproc)"
free -h
df -h .

# 运行Python脚本 (移除nice -n -10避免权限问题)
echo "开始运行优化的SVM训练..."
time python advanced_svm.py

# 完成后显示结果文件
echo "训练完成，查看结果文件:"
ls -lh ./output/

echo "执行完毕!"