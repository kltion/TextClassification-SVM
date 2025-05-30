# 数据集下载链接

由于GitHub对文件大小有限制，数据集文件未能直接上传到仓库中。请从以下链接获取：

## 阿里天池比赛原始数据下载

您可以直接从阿里天池比赛官网下载原始数据集：
https://tianchi.aliyun.com/competition/entrance/531810/information

## 数据集说明

- `train_set.csv` - 训练数据集（约839MB）
- `test_a.csv` - 测试数据集（约210MB）

## 使用方法

1. 下载数据集文件
2. 将文件放置到项目的`input`目录中
3. 确保文件名称为`train_set.csv`和`test_a.csv`
4. 运行`run_cpu_optimized.sh`脚本开始训练和评估

## 数据格式

数据集为CSV格式，以Tab（\t）分隔：
- 训练集：包含label和text两个字段
- 测试集：包含text字段