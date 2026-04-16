# SEMixer 论文复现详细操作步骤

本文档严格依据官方仓库 [Meteor-Stars/SEMixer](https://github.com/Meteor-Stars/SEMixer) 与论文《SEMixer: Semantics Enhanced MLP-Mixer for Multiscale Mixing and Long-term Time Series Forecasting》整理，不补写未在仓库或论文中明确给出的版本号、数值和结果。

## 1. 复现目标

目标是复现论文在 10 个公开长序列预测数据集上的主实验，以及论文附录中的更长预测步长实验。官方代码主入口是 `run.py`，训练流程位于 `exp/exp_main.py`，数据处理位于 `data_provider/`，模型定义位于 `models/`。

## 2. 环境准备

论文正文明确写到实验在 NVIDIA GeForce RTX 3090 GPU 上、基于 PyTorch 完成；官方仓库未固定 Python/CUDA/PyTorch 的精确版本。因此建议按官方代码导入依赖后，在本地环境中确认是否可直接运行。

建议先做两件事：

1. 使用你现有的 conda 环境 `daris-research`，如果其中已经安装了 `torch`，优先在该环境里验证。
2. 只补装仓库缺失的依赖，不擅自改写论文未注明的版本号。

仓库中可见的关键依赖包括：`torch`、`numpy`、`pandas`、`scipy`、`scikit-learn`、`matplotlib`、`einops`、`transformers`、`ptflops`、`fvcore`、`requests`。

## 3. 数据集准备

论文实验使用 10 个公开数据集：ETTh1、ETTh2、ETTm1、ETTm2、Weather、Electricity、Traffic、Exchange、Solar Energy、ILI。

论文附录给出的数据统计如下：

- ETTh1/ETTh2：7 维，(8545, 2881, 2881)，Hourly
- ETTm1/ETTm2：7 维，(34465, 11521, 11521)，15min
- Weather：21 维，(36792, 5271, 10540)，10min
- Exchange rate：9 维，(5120, 665, 1422)，Daily
- Electricity：321 维，(18317, 2633, 5261)，Hourly
- Solar energy：137 维，(36601, 5161, 10417)，10min
- ILI：7 维，(617, 74, 170)，Daily
- Traffic：862 维，(12185, 1757, 3509)，Hourly

官方仓库提供了 `utils/download_data.py`，用于下载数据包。数据加载逻辑在 `data_provider/data_loader.py`：

- ETTh1/ETTh2 使用 `Dataset_ETT_hour`
- ETTm1/ETTm2 使用 `Dataset_ETT_minute`
- 其他自定义数据使用 `Dataset_Custom`
- `pred` 模式使用 `Dataset_Pred`

注意：

- ETT 系列采用仓库内固定切分边界
- 自定义数据集采用 70/20/10 切分
- 标准化只在训练集上拟合
- 时间特征由日期列生成

## 4. 代码获取与安装

先获取官方仓库：

```bash
git clone https://github.com/Meteor-Stars/SEMixer.git
cd SEMixer
```

如果仓库已有环境说明文件，优先按仓库说明安装。若缺少依赖，可按导入错误逐个补装。不要为了“凑能跑”而修改模型超参数。

## 5. 训练入口与总体流程

主入口是 `run.py`。脚本会：

1. 解析参数
2. 按数据集和预测长度设置超参数
3. 循环多个随机种子
4. 调用 `Exp_Main.train()` 和 `Exp_Main.test_inference_time()`
5. 生成 `record_args.json`、checkpoint、日志与测试输出

论文与代码中一致的核心训练设置：

- seeds：0, 1, 2, 3, 4
- 预测长度：96, 192, 336, 720
- 优化器：Adam
- 损失：MSELoss
- 学习率调度：OneCycleLR
- 早停：EarlyStopping
- 训练轮数：30，Traffic 为 50
- 评估指标：MAE、MSE，代码还返回 RMSE、MAPE、MSPE、RSE、CORR

## 6. SEMixer 的论文级超参数

论文正文 4.1.3 与官方脚本共同给出的 SEMixer 核心超参数如下：

- 维度 D：128
- temporal mixing 中的线性融合维度：64
- 最细尺度 patch 数 N1：64
- scale 数 S：4
- scale factors：2, 4, 8
- random disconnection probability p：0.85
- 训练 epoch：30，Traffic 为 50

仓库中 SEMixer 相关默认还包括：

- `reduce_dim=64`
- `maximum_patch_num=64`
- `multi_scale=True`
- `scale_factors=[1, 2, 4, 8]`
- `sample_num=5`
- `connection_probability=0.85`
- `Random_Attention_Mechanism=True`
- `self_attn=False`
- `prob_attn=False`

## 7. 数据集级别的官方配置

`run.py` 中对不同数据集有显式分支，以下是从官方脚本恢复的关键设置。这里仅列出代码中明确出现的值。

### 7.1 ETTh1

- `seq_len`：512 / 1024 / 1536 / 1664 / 2048 / 2560，按预测长度与输入长度搜索策略分支变化
- `enc_in=7`
- `n_heads=4`
- `d_model=128`
- `d_ff=128`
- `dropout=0.3`
- `fc_dropout=0.3`
- `batch_size=128`
- `learning_rate=0.0001`

### 7.2 ETTh2

- `enc_in=7`
- `n_heads=4`
- `d_ff=128`
- `dropout=0.3`
- `fc_dropout=0.3`
- `batch_size=128`
- `seq_len`：代码分支中出现 1280、1024、768 等取值，取决于预测长度

### 7.3 ETTm1

- `enc_in=7`
- `n_heads=16`
- `d_ff=256`
- `dropout=0.2`
- `fc_dropout=0.2`
- `batch_size=128`
- `seq_len`：代码分支中出现 768、1536、1664 等取值

### 7.4 ETTm2

- `enc_in=7`
- `n_heads=16`
- `d_ff=256`
- `dropout=0.2`
- `fc_dropout=0.2`
- `batch_size=128`
- `seq_len`：代码分支中出现 768、1664 等取值

### 7.5 Weather

- `enc_in=21`
- `n_heads=16`
- `d_ff=256`
- `dropout=0.2`
- `fc_dropout=0.2`
- `batch_size=64`
- `seq_len=2048`

### 7.6 Electricity

- `enc_in=321`
- `n_heads=16`
- `d_ff=256`
- `dropout=0.2`
- `fc_dropout=0.2`
- `batch_size=32`
- `seq_len=1664`
- 代码后续还开启了 `var_decomp=True` 与 `var_sp_num=15`

### 7.7 Traffic

- `enc_in=862`
- `n_heads=16`
- `d_model=128`
- `d_ff=256`
- `dropout=0.2`
- `fc_dropout=0.2`
- `batch_size=12`
- `learning_rate=0.0001`
- `seq_len=2048`

### 7.8 Exchange rate

- `enc_in=8`
- `n_heads=4`
- `d_model=128`
- `d_ff=128`
- `dropout=0.3`
- `fc_dropout=0.3`
- `batch_size=16`
- `learning_rate=0.0025`
- `e_layers=3`
- `seq_len=512`

### 7.9 Solar_AL

- `enc_in=137`
- `n_heads=4`
- `d_model=128`
- `d_ff=128`
- `dropout=0.3`
- `fc_dropout=0.3`
- `batch_size=64`

### 7.10 national_illness / ILI

- `enc_in=7`
- `batch_size=12`

## 8. 推荐复现命令

官方脚本没有把所有数据集分支整理成单独命令文件，因此最稳妥的方式是直接按 `run.py` 的分支运行。下面给出的是执行思路，不是额外发明的新配置。

### 8.1 标准长预测实验

对每个数据集、每个预测长度分别运行 `run.py`。主实验通常使用：

- `pred_len=96`

## 9. 本地可运行修复记录

在本次会话的 Windows 环境中，官方仓库需要做少量兼容性修复才能稳定跑通：

1. `utils/tools.py` 中的 `np.Inf` 需要改成 `np.inf`，否则会被当前 NumPy 版本直接报错。
2. `exp/exp_main.py` 里原本会导入一批可选模型，其中 `DeformableTST` 依赖的 `layers.Global_Attn` 在仓库中不存在，导致 SEMixer 训练在导入阶段失败。实际复现 SEMixer 时只保留 SEMixer 的导入即可。
3. `run.py` 里需要补上 `scaleformers`、`Self_Attention_Mechanism`、`test` 等旧字段，并将 `num_workers` 设为 0，避免 Windows 上的 DataLoader worker 问题。

这些修复不改变 SEMixer 的核心算法，只是让官方脚本在当前 Windows + NumPy 2 + CUDA PyTorch 环境中可执行。

## 10. 已验证的本地运行结果

本次会话已经成功跑通 ETTh1 的一个官方配置：`seed=0`、`pred_len=96`。训练过程完成 30 个 epoch，并保存了最佳 checkpoint。单独加载该 checkpoint 后，在测试集上得到：

- MAE: 0.4000023603439331
- MSE: 0.36982405185699463
- RMSE: 0.6081315875053406
- MAPE: 9.441551208496094
- MSPE: 35191.19921875
- RSE: 0.5776375532150269

对应输出目录为 `LongTermTSF_SEMixer/ETTh1/random_seed_0/ETTh1_SEMixer_SeqLen1280_PredLen96_HiddenDim_128/`。
- `pred_len=192`
- `pred_len=336`
- `pred_len=720`

如果你需要与论文主表一致，必须按官方脚本的数据集分支和输入长度搜索逻辑执行，而不是固定一个通用输入长度。

### 8.2 更长预测实验

论文附录还给出：

- `pred_len=1020`
- `pred_len=1320`
- `pred_len=1620`

这部分对应固定输入长度 2560 或 2048 的设置，具体取法以论文 Table 9 / Table 11 为准。

## 9. 训练与测试流程

官方训练流程在 `exp/exp_main.py` 中：

1. `train()` 负责训练、验证、早停和保存 `checkpoint.pth`
2. `vali()` 计算验证集指标
3. `test_inference_time()` 载入 checkpoint，测量推理耗时，并保存预测结果和真值

验证阶段使用的指标函数返回：MAE、MSE、RMSE、MAPE、MSPE、RSE、CORR。

## 10. 结果整理建议

建议将每次运行输出整理成三类文件：

- 配置记录：`record_args.json`
- 模型权重：`checkpoint.pth`
- 预测与真值：`test_results/` 下的结果文件

论文主表与附录表对应关系如下：

- 主表：96 / 192 / 336 / 720 的平均结果
- 附录表：每个预测长度的完整结果
- 更长预测实验：1020 / 1320 / 1620

## 11. 需要特别注意的复现约束

1. 不要擅自修改论文未声明的训练轮数、随机种子、学习率和输入长度。
2. 不要把代码默认值当成论文值，尤其是不同数据集分支里的 `seq_len`、`batch_size`、`n_heads`、`d_ff`。
3. 不要把未验证的环境版本写成“已确认”。
4. 不要把本地没有跑出的分数写进复现报告。

## 12. 本次会话状态说明

本次会话只做了代码与论文对照分析，没有实际跑完训练或测试，因此这里不提供“本次复现结果数值”。如果你要真正得到结果，需要按上述分支逐个运行并收集输出。

## 13. 最小可执行检查清单

- 数据包已下载并解压到仓库约定位置
- `torch` 可导入
- `run.py` 可启动
- 至少能成功跑通一个小数据集和一个预测长度
- checkpoint 能保存，`test_inference_time()` 能完成
- 结果目录里能产出预测值和真值文件

完成以上后，再按论文表格逐项整理即可。