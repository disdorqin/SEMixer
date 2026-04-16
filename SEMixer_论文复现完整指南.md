# SEMixer 论文复现完整指南

> 本文档严格基于官方仓库 [Meteor-Stars/SEMixer](https://github.com/Meteor-Stars/SEMixer) 与论文《SEMixer: Semantics Enhanced MLP-Mixer for Multiscale Mixing and Long-term Time Series Forecasting》整理，所有参数和配置均来自代码和论文的明确内容。

---

## 目录

1. [环境配置](#1-环境配置)
2. [数据集准备](#2-数据集准备)
3. [模型核心参数](#3-模型核心参数)
4. [各数据集详细配置](#4-各数据集详细配置)
5. [训练命令](#5-训练命令)
6. [结果验证](#6-结果验证)

---

## 1. 环境配置

### 1.1 已验证的环境版本（.venv）

```
Python: 3.11.9
torch: 2.5.1+cu121 (CUDA 12.1)
numpy: 2.4.4
pandas: 3.0.2
scipy: 1.17.1
scikit-learn: 1.8.0
matplotlib: 3.10.8
einops: 0.8.2
transformers: 5.5.4
ptflops: 0.7.5
fvcore: 0.1.5.post20221221
requests: 2.33.1
```

### 1.2 激活环境并验证

```bash
# Windows
.venv\Scripts\activate

# 验证环境
python --version
python -c "import torch; print(torch.__version__)"
python -c "import numpy; print(numpy.__version__)"
```

### 1.3 论文要求对比

| 项目 | 论文要求 | 当前环境 | 是否一致 |
|------|----------|----------|----------|
| GPU | NVIDIA GeForce RTX 3090 | 需用户确认 | 需确认 |
| PyTorch | PyTorch 框架 | 2.5.1+cu121 | ✅ 兼容 |
| 核心依赖 | torch, numpy, pandas, scipy, sklearn, matplotlib, einops | 已安装 | ✅ 一致 |

---

## 2. 数据集准备

### 2.1 数据集统计（论文 Table 6）

| 数据集 | 变量数 | 训练/验证/测试 | 频率 |
|--------|--------|----------------|------|
| ETTh1 | 7 | 8545 / 2881 / 2881 | 小时 |
| ETTh2 | 7 | 8545 / 2881 / 2881 | 小时 |
| ETTm1 | 7 | 34465 / 11521 / 11521 | 15 分钟 |
| ETTm2 | 7 | 34465 / 11521 / 11521 | 15 分钟 |
| Weather | 21 | 36792 / 5271 / 10540 | 10 分钟 |
| Electricity | 321 | 18317 / 2633 / 5261 | 小时 |
| Traffic | 862 | 12185 / 1757 / 3509 | 小时 |
| Exchange | 9 | 5120 / 665 / 1422 | 天 |
| Solar Energy | 137 | 36601 / 5161 / 10417 | 10 分钟 |
| ILI | 7 | 617 / 74 / 170 | 天 |

### 2.2 数据集位置

数据已存在于 `repo/dataset/` 目录下：
- `ETTh1.csv`
- `ETTh2.csv`
- `ETTm1.csv`
- `ETTm2.csv`
- `weather.csv`
- `electricity.csv`
- `traffic.csv`
- `exchange_rate.csv`
- `solar_AL.txt`
- `national_illness.csv`

### 2.3 数据切分规则

**ETT 系列（ETTh1, ETTh2, ETTm1, ETTm2）：**
- 训练集：前 12 个月
- 验证集：接下来 4 个月
- 测试集：最后 8 个月

**其他数据集（自定义 Dataset_Custom）：**
- 训练集：70%
- 验证集：20%
- 测试集：10%

---

## 3. 模型核心参数

### 3.1 SEMixer 核心架构参数（论文 4.1.3）

```python
# 核心参数
d_model = 128              # 论文记为 D，patch/position embedding hidden size
reduce_dim = 64            # temporal mixing 线性融合维度
maximum_patch_num = 64     # 最细尺度 patch 数 N1
scale_factors = [1, 2, 4, 8]  # scale 数 S=4，scale factors 为 2, 4, 8
sample_num = 5             # 采样数
connection_probability = 0.85  # random disconnection probability p
eib_num = 1                # 多尺度层数
eib_num_1scale = 1         # 第一尺度层数
```

### 3.2 RAM（Random Attention Mechanism）参数

```python
Random_Attention_Mechanism = True   # 启用 RAM
self_attn = False                   # 不使用标准 self-attention
prob_attn = False                   # 不使用 ProbSparse attention
```

### 3.3 训练设置

```python
seeds = [0, 1, 2, 3, 4]      # 5 个随机种子
pred_len = [96, 192, 336, 720]  # 预测长度
train_epochs = 30            # 训练轮数（Traffic 为 50）
batch_size = 按数据集配置      # 见下表
learning_rate = 按数据集配置  # 见下表
optimizer = Adam             # 优化器
loss = MSELoss               # 损失函数
lr_scheduler = OneCycleLR    # 学习率调度
early_stopping_patience = 100 # 早停耐心值
```

---

## 4. 各数据集详细配置

### 4.1 ETTh1

```python
data_type = 'ETTh1'
data_path = 'ETTh1.csv'
enc_in = 7          # 输入变量数
n_heads = 4
d_ff = 128
dropout = 0.3
fc_dropout = 0.3
head_dropout = 0
batch_size = 128
learning_rate = 0.0001

# seq_len 根据 pred_len 自动设置：
# pred_len = 96 或 192:  seq_len = 1280
# pred_len = 336 或 720: seq_len = 384
# pred_len > 720:       seq_len = 2560
```

### 4.2 ETTh2

```python
data_type = 'ETTh2'
data_path = 'ETTh2.csv'
enc_in = 7
n_heads = 4
d_ff = 128
dropout = 0.3
fc_dropout = 0.3
head_dropout = 0
batch_size = 128

# seq_len 根据 pred_len 自动设置：
# pred_len = 96:  seq_len = 1280
# pred_len = 192 或 336: seq_len = 1024
# pred_len = 720:      seq_len = 768
# pred_len > 720:      seq_len = 2560
```

### 4.3 ETTm1

```python
data_type = 'ETTm1'
data_path = 'ETTm1.csv'
enc_in = 7
n_heads = 16
d_ff = 256
dropout = 0.2
fc_dropout = 0.2
head_dropout = 0
batch_size = 128

# seq_len 根据 pred_len 自动设置：
# pred_len = 96:  seq_len = 768
# pred_len = 192 或 336: seq_len = 1536
# pred_len = 720:      seq_len = 1664
# pred_len > 720:      seq_len = 2560
```

### 4.4 ETTm2

```python
data_type = 'ETTm2'
data_path = 'ETTm2.csv'
enc_in = 7
n_heads = 16
d_ff = 256
dropout = 0.2
fc_dropout = 0.2
head_dropout = 0
batch_size = 128

# seq_len 根据 pred_len 自动设置：
# pred_len = 96:  seq_len = 768
# pred_len = 192 或 336 或 720: seq_len = 1664
# pred_len > 720:      seq_len = 2560
```

### 4.5 Weather

```python
data_type = 'weather'
data_path = 'weather.csv'
enc_in = 21
n_heads = 16
d_ff = 256
dropout = 0.2
fc_dropout = 0.2
head_dropout = 0
batch_size = 64

# seq_len = 2048 (所有 pred_len)
```

### 4.6 Electricity

```python
data_type = 'electricity'
data_path = 'electricity.csv'
enc_in = 321
n_heads = 16
d_ff = 256
dropout = 0.2
fc_dropout = 0.2
head_dropout = 0
batch_size = 32
var_decomp = True      # 启用变量分解
var_sp_num = 15        # 变量分组数

# seq_len = 1664 (所有 pred_len)
```

### 4.7 Traffic

```python
data_type = 'traffic'
data_path = 'traffic.csv'
enc_in = 862
n_heads = 16
d_model = 128
d_ff = 256
dropout = 0.2
fc_dropout = 0.2
head_dropout = 0
batch_size = 12        # 24//2
learning_rate = 0.0001
train_epochs = 50      # Traffic 专用，其他数据集为 30

# seq_len = 2048 (所有 pred_len)
```

### 4.8 Exchange Rate

```python
data_type = 'exchange_rate'
data_path = 'exchange_rate.csv'
enc_in = 8
n_heads = 4
d_model = 128
d_ff = 128
dropout = 0.3
fc_dropout = 0.3
head_dropout = 0
batch_size = 32
learning_rate = 0.0025
e_layers = 3

# seq_len = 512 (所有 pred_len)
```

### 4.9 Solar Energy

```python
data_type = 'solar_AL'
data_path = 'solar_AL.txt'
enc_in = 137
n_heads = 4
d_model = 128
d_ff = 128
dropout = 0.3
fc_dropout = 0.3
head_dropout = 0
batch_size = 64
learning_rate = 0.0025
e_layers = 3

# seq_len = 512 (所有 pred_len)
```

### 4.10 ILI (national_illness)

```python
data_type = 'national_illness'
data_path = 'national_illness.csv'
enc_in = 7
n_heads = 4
d_model = 16
d_ff = 128
dropout = 0.3
fc_dropout = 0.3
head_dropout = 0
batch_size = 16
learning_rate = 0.0025
e_layers = 3

# seq_len = 512 (所有 pred_len)
```

---

## 5. 训练命令

### 5.1 运行全部实验（10 个数据集 × 4 个预测长度 × 5 个种子 = 200 次训练）

```bash
cd repo
.venv\Scripts\python.exe run.py
```

### 5.2 修改特定数据集和预测长度

编辑 `run.py` 文件第 419-423 行：

```python
if __name__ == "__main__":
    Seeds_All = [0, 1, 2, 3, 4]
    pred_len = [96, 192, 336, 720]  # 修改这里
    # 或者修改 main() 调用中的数据集类型
```

### 5.3 单数据集单预测长度测试

在 `run.py` 的 `main()` 函数中修改：

```python
# 修改第 103 行
args.data_type = 'ETTh1'  # 改为其他数据集

# 修改第 108 行
args.pred_len = 96  # 改为其他预测长度
```

然后运行：
```bash
.venv\Scripts\python.exe -c "from run import main; main(0, 96)"
```

### 5.4 更长预测实验（论文附录）

```python
pred_len = [1020, 1320, 1620]
# 此时 seq_len 会自动设置为 2560
```

---

## 6. 结果验证

### 6.1 输出目录结构

训练完成后，结果保存在以下目录：

```
LongTermTSF_SEMixer/
└── {dataset}/
    └── random_seed_{seed}/
        └── {dataset}_SEMixer_SeqLen{seq_len}_PredLen{pred_len}_HiddenDim_128/
            ├── record_args.json       # 实验配置
            ├── checkpoint.pth         # 最佳模型权重
            ├── record_all_loss_train.json  # 训练损失
            ├── record_all_loss_val.json    # 验证损失
            └── record_all_loss_test.json   # 测试损失
```

### 6.2 测试输出

测试完成后，预测结果保存在：

```
test_results/
└── {dataset}_SEMixer_SeqLen{seq_len}_PredLen{pred_len}_HiddenDim_128/
    ├── pred.npy    # 预测值
    └── true.npy    # 真实值
```

### 6.3 评估指标

代码输出的指标包括：
- **MSE**: Mean Squared Error（均方误差）
- **MAE**: Mean Absolute Error（平均绝对误差）
- **RMSE**: Root Mean Squared Error（均方根误差）
- **MAPE**: Mean Absolute Percentage Error（平均绝对百分比误差）
- **MSPE**: Mean Squared Percentage Error（均方百分比误差）
- **RSE**: Relative Squared Error（相对平方误差）
- **CORR**: Correlation Coefficient（相关系数）

### 6.4 论文主表结果参考（Table 1）

| 数据集 | MSE | MAE |
|--------|-----|-----|
| ETTh1 | 0.400 | 0.418 |
| ETTh2 | 0.331 | 0.382 |
| ETTm1 | 0.342 | 0.375 |
| ETTm2 | 0.241 | 0.312 |
| Weather | 0.216 | 0.258 |
| Electricity | 0.154 | 0.249 |
| ILI | 2.385 | 1.080 |
| Exchange | 0.344 | 0.397 |
| Solar Energy | 0.184 | 0.245 |
| Traffic | 0.388 | 0.268 |

### 6.5 更长预测结果参考（Table 2）

| 数据集 | pred_len | MSE | MAE |
|--------|----------|-----|-----|
| ETTh1 | 1020 | 0.533 | 0.520 |
| ETTh1 | 1320 | 0.593 | 0.552 |
| ETTh1 | 1620 | 0.661 | 0.591 |
| ETTh2 | 1020 | 0.452 | 0.483 |
| ETTh2 | 1320 | 0.499 | 0.511 |
| ETTh2 | 1620 | 0.547 | 0.538 |

---

## 7. 常见问题排查

### 7.1 CUDA 相关

```bash
# 检查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"
```

### 7.2 内存不足

如果遇到 CUDA 内存不足：
- 减小 `batch_size`
- 缩短 `seq_len`

### 7.3 Windows DataLoader 问题

代码已设置 `num_workers = 0` 以避免 Windows 多进程问题。

---

## 8. 复现检查清单

- [ ] 环境已激活（.venv）
- [ ] 数据集已下载到 `repo/dataset/`
- [ ] GPU 可用（nvidia-smi 确认）
- [ ] 运行 `run.py` 开始训练
- [ ] 检查 `LongTermTSF_SEMixer/` 目录输出
- [ ] 记录每个数据集 5 个种子的平均 MSE/MAE
- [ ] 与论文 Table 1 对比结果

---

## 9. 执行脚本示例

创建 `run_reproduction.bat` 批量运行：

```batch
@echo off
cd /d %~dp0repo
call ..\.venv\Scripts\activate

echo Starting SEMixer reproduction...
echo Running ETTh1 with pred_len=96, seeds 0-4

for %%s in (0 1 2 3 4) do (
    echo Running seed %%s
    ..\.venv\Scripts\python.exe -c "from run import main; main(%%s, 96)"
)

echo Reproduction complete!
```

---

**文档版本**: 1.0  
**最后更新**: 2024  
**基于代码版本**: Meteor-Stars/SEMixer 官方仓库