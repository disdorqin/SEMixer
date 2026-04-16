# SEMixer 完整实验报告

本文档严格依据官方仓库 [Meteor-Stars/SEMixer](https://github.com/Meteor-Stars/SEMixer) 与论文《SEMixer: Semantics Enhanced MLP-Mixer for Multiscale Mixing and Long-term Time Series Forecasting》整理，不补写未在论文和仓库中明确给出的版本号、结果和超参数。

## 1. 论文摘要

SEMixer 面向长序列时间序列预测中的多尺度建模问题，提出了一个轻量级多尺度框架。作者指出，时间序列中存在冗余、噪声以及非相邻尺度之间的语义鸿沟，导致多尺度依赖的高效对齐和融合困难。为此，论文提出两个核心模块：Random Attention Mechanism，简称 RAM，以及 Multiscale Progressive Mixing Chain，简称 MPMC。

RAM 通过训练阶段的随机时间块交互增强 patch 语义，并在推理阶段用 dropout 风格的集成近似这些交互；MPMC 则以更节省显存的方式分层堆叠 RAM 和 MLP-Mixer，在保持效率的同时改进跨尺度混合。论文在 10 个公开数据集和 2025 CCF AlOps Challenge 上验证了方法效果，结果显示 SEMixer 在精度与效率上均优于大量基线。

## 2. 实验环境

论文正文明确说明实验在 NVIDIA GeForce RTX 3090 上完成，训练框架为 PyTorch。官方仓库未给出固定的 Python、CUDA 或 PyTorch 小版本，因此本报告只记录论文与仓库明确写出的环境信息。

### 2.1 官方代码结构

- `run.py`：实验入口与数据集超参数分支
- `exp/exp_main.py`：训练、验证、测试、推理耗时统计
- `data_provider/`：数据集切分、归一化和 DataLoader
- `models/SEMixer.py`：SEMixer 主模型
- `utils/metrics.py`：指标计算
- `utils/tools.py`：早停、学习率调度、可视化、参数量/FLOPs 辅助函数

### 2.2 统一训练设置

论文与官方代码一致的关键设置：

- seeds：0, 1, 2, 3, 4
- 预测长度：96, 192, 336, 720
- 训练 epoch：30，Traffic 为 50
- 优化器：Adam
- 损失函数：MSE
- 学习率调度：OneCycleLR
- 早停：EarlyStopping
- 评估指标：MSE、MAE；代码还计算 RMSE、MAPE、MSPE、RSE、CORR

### 2.3 SEMixer 关键超参数

论文 4.1.3 给出的核心设置如下：

- patch/position embedding hidden size D = 128
- temporal mixing 线性融合维度 = 64
- finest scale patch 数 N1 = 64
- scale 数 S = 4
- scale factors = 2, 4, 8
- sampling disconnection probability p = 0.85

官方仓库中的对应实现还包含：

- `reduce_dim=64`
- `maximum_patch_num=64`
- `multi_scale=True`
- `scale_factors=[1, 2, 4, 8]`
- `sample_num=5`
- `connection_probability=0.85`

## 3. 数据集说明

论文验证了 10 个公共数据集：ETTh1、ETTh2、ETTm1、ETTm2、Weather、Electricity、Traffic、Exchange、Solar Energy、ILI。

### 3.1 数据集统计

论文附录 Table 6 给出以下统计信息：

| 数据集 | 维度 | 训练/验证/测试 | 频率 |
| --- | --- | --- | --- |
| ETTh1, ETTh2 | 7 | (8545, 2881, 2881) | Hourly |
| ETTm1, ETTm2 | 7 | (34465, 11521, 11521) | 15min |
| Weather | 21 | (36792, 5271, 10540) | 10min |
| Exchange rate | 9 | (5120, 665, 1422) | Daily |
| Electricity | 321 | (18317, 2633, 5261) | Hourly |
| Solar energy | 137 | (36601, 5161, 10417) | 10min |
| ILI | 7 | (617, 74, 170) | Daily |
| Traffic | 862 | (12185, 1757, 3509) | Hourly |

### 3.2 数据来源与切分逻辑

官方代码中：

- ETTh1/ETTh2 使用小时级 ETT 数据集类
- ETTm1/ETTm2 使用分钟级 ETT 数据集类
- 其余公开数据集使用通用自定义数据集类
- 自定义数据集默认 70/20/10 切分
- 验证和测试阶段使用训练集拟合的 scaler

## 4. 方法概述

SEMixer 是一个多尺度 MLP-Mixer 风格框架，核心思想是先做多尺度 patch 化，再通过 RAM 提升 patch 语义，最后通过 MPMC 逐级融合相邻尺度信息。

### 4.1 Multiscale Encoding Block

输入历史序列先做 Instance Normalization，然后在多个尺度上 patchify。最细尺度使用较短 patch 和较小 stride，较粗尺度按 scale factor 放大 patch 长度和 stride。这样可同时捕获局部波动和全局趋势。

### 4.2 RAM

RAM 用 Bernoulli 随机掩码替代标准 attention 中的 Q-K-V 计算。训练时采样大量随机交互矩阵，推理时用 dropout 式缩放近似集成效果。论文强调 RAM 的目标不是显式学习注意力权重，而是用大量随机交互增强 patch 语义，减少冗余与噪声影响。

### 4.3 MPMC

MPMC 按最细到最粗的顺序逐级串联相邻尺度，把 RAM 与 MLP-Mixer 组合成 temporal mixing block，再将各尺度输出整合到最终预测头。与直接拼接所有尺度相比，MPMC 更省显存，也更能缓解非相邻尺度的语义差异。

## 5. 主实验结果

论文主表 Table 1 报告了 96、192、336、720 四个预测长度的平均结果，表 2 报告了更长输入和更长预测长度的设置，表 3 报告了 2025 CCF AlOps Challenge 的结果。

### 5.1 Table 1：公开数据集主结果

论文在主表中报告 SEMixer 与 11 个基线的平均 MSE / MAE。下面列出论文中可直接确认的 SEMixer 与关键基线对比摘要。完整表格很长，这里保留论文正文最核心的结论与关键数值。

- ETTh1：SEMixer 0.400 / 0.418
- ETTh2：SEMixer 0.331 / 0.382
- ETTm1：SEMixer 0.342 / 0.375
- ETTm2：SEMixer 0.241 / 0.312
- Weather：SEMixer 0.216 / 0.258
- Electricity：SEMixer 0.154 / 0.249
- ILI：SEMixer 2.385 / 1.080
- Exchange：SEMixer 0.344 / 0.397
- Solar Energy：SEMixer 0.184 / 0.245
- Traffic：SEMixer 0.388 / 0.268

论文正文总结为：SEMixer 在这些数据集上相对已有方法取得了约 5% 到 15% 的 MSE 改善。

### 5.2 Table 2：更长预测长度

Table 2 使用输入长度 2560，预测 1020、1320、1620 步，并与固定输入长度 2048 的设置一起评估。论文结论是，SEMixer 在更长预测任务上仍然保持明显优势。

论文中可直接确认的 SEMixer 结果包括：

- ETTh1：1020 为 0.533 / 0.520，1320 为 0.593 / 0.552，1620 为 0.661 / 0.591
- ETTh2：1020 为 0.452 / 0.483，1320 为 0.499 / 0.511，1620 为 0.547 / 0.538
- ETTm1：1020 为 0.415 / 0.422，1320 为 0.427 / 0.431，1620 为 0.433 / 0.437
- ETTm2：1020 为 0.353 / 0.393，1320 为 0.360 / 0.401，1620 为 0.353 / 0.399
- Weather：1020 为 0.312 / 0.342，1320 为 0.323 / 0.349，1620 为 0.337 / 0.360

### 5.3 Table 3：2025 CCF AlOps Challenge

论文报告 SEMixer 在该无线网络 KPI 预测任务上取得了第三名，并在对比中给出如下预测误差：

| 模型 | 误差 |
| --- | --- |
| SEMixer | 0.4425 |
| Deform.TST | 0.4491 |
| TimeXer | 0.4486 |
| ModernTCN | 0.4482 |
| PatchTST | 0.4506 |
| TimeMixer | 0.4479 |

论文文字说明 SEMixer 在该任务中是参赛模型里表现最优的。

## 6. 消融实验

### 6.1 Table 4：RAM 相关消融

论文将标准 self-attention、ProbSparse、AutoCorrelation、FourierAttention、LogSparse、LSH、Performer 等机制与 RAM 对比。结论是：替换为标准注意力机制会带来性能下降和更高开销，而 RAM 在效率和效果之间更均衡。

论文正文明确指出，在 ETTh1、ETTh2、ETTm1、ETTm2、Weather 上，去掉 RAM 或用标准注意力替换后平均结果会变差，且标准注意力仍需 Q-K-V 计算，开销更高。

### 6.2 Table 5：MPMC 与噪声鲁棒性

论文在 Table 5 中报告了更长输入和更长预测长度下的平均结果，以及噪声注入场景。可确认的关键结论如下：

- 去掉 MPMC 会显著提高误差
- 去掉 RAM 也会提高误差，但幅度小于去掉 MPMC 的极端情况
- 加入噪声后，SEMixer 仍保持更好的鲁棒性

论文给出的部分数值包括：

- w/o noise 的 SEMixer：ETTh1 0.596 / 0.554，ETTh2 0.551 / 0.536，ETTm1 0.425 / 0.430，ETTm2 0.355 / 0.398，Weather 0.323 / 0.349
- w/o MPMC：ETTh1 0.635 / 0.577，ETTh2 0.585 / 0.555，ETTm1 0.430 / 0.433，ETTm2 0.365 / 0.405，Weather 0.327 / 0.352
- w/ SAM：ETTh1 0.638 / 0.579，ETTh2 0.567 / 0.546，ETTm1 0.433 / 0.437，ETTm2 0.359 / 0.402，Weather 0.326 / 0.351
- w/o RAM：ETTh1 0.612 / 0.562，ETTh2 0.568 / 0.546，ETTm1 0.427 / 0.432，ETTm2 0.373 / 0.412，Weather 0.334 / 0.362

### 6.3 Table 7：MPMC 结构验证

附录 Table 7 验证了不同多尺度结构的效果。论文结论是：

- 只用单尺度输入最差
- 只做同尺度内部混合次之
- 加入跨尺度 progressive mixing 后最好

论文附录说明 Patch count 比例从最细到最粗为 8:4:2:1，进一步支持 MPMC 的分层设计。

## 7. 效率分析

论文附录 A.2.2 给出 RAM 的时间复杂度分析，并给出 FLOPs 对比：输入大小 [1,1024,321] 下，PatchSTT、TimeMixer、TSMixer、SEMixer 的 FLOPs 分别为 2.8512、5.6810、8.559、4.679。

这部分说明 SEMixer 在维持精度的同时，计算开销低于部分基线。

## 8. 论文结论

论文的核心结论是：

1. RAM 可以通过随机交互增强 patch 语义，并用 dropout 风格集成近似推理阶段的多交互效果。
2. MPMC 可以在控制显存和计算开销的同时建立有效的跨尺度依赖。
3. 在 10 个公开数据集和 2025 CCF AlOps Challenge 上，SEMixer 都展示了稳定的精度和效率优势。

## 9. 代码实现与论文一致性说明

官方仓库与论文在以下方面是对齐的：

- 训练 seeds 为 0 到 4
- 主评估长度为 96 / 192 / 336 / 720
- 论文级超参数与代码中的 SEMixer 分支一致，包括 D=128、reduce_dim=64、S=4、scale factors=2/4/8、p=0.85
- 数据集加载与切分逻辑和论文附录表 6 相吻合

需要注意的是，官方脚本里还存在按数据集分支设置的 `seq_len`、`batch_size`、`n_heads` 等参数，这些必须保留，否则无法保证和论文一致。

## 10. 本次会话实测

本次会话已经在本地环境中完成了一个官方配置的实际训练与测试评估：ETTh1、seed 0、pred_len 96。为保证 Windows 环境可运行，已修复几个官方仓库里的兼容性问题，包括：

- 将 `utils/tools.py` 中的 `np.Inf` 改为 `np.inf`
- 收缩 `exp/exp_main.py` 的模型导入范围，避免缺失的可选模块阻塞 SEMixer
- 补齐 `run.py` 中缺失的旧参数字段，并将 `num_workers` 设为 0 以避免 Windows DataLoader worker 问题

该次运行的最佳 checkpoint 测试指标为：

- MAE: 0.4000023603439331
- MSE: 0.36982405185699463
- RMSE: 0.6081315875053406
- MAPE: 9.441551208496094
- MSPE: 35191.19921875
- RSE: 0.5776375532150269

对应的训练日志显示，验证集最佳值出现在较早的 epoch，且 checkpoint 已成功保存到 `LongTermTSF_SEMixer/ETTh1/random_seed_0/ETTh1_SEMixer_SeqLen1280_PredLen96_HiddenDim_128/`。

未完成部分仍然是其余数据集与预测长度的全量网格，这些尚未在本会话内全部执行完毕。

## 11. 结论摘要

SEMixer 的贡献不是单纯增加模型容量，而是通过随机交互增强语义、通过分层多尺度链条降低跨尺度噪声，从而在长序列预测任务中同时改善精度和效率。论文证据链完整，官方代码也提供了相应的训练入口、数据加载与评估逻辑，具备可操作的复现基础。