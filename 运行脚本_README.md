# SEMixer 运行脚本说明

## 数据集难度分类

| 难度 | 数据集 | 变量数 | 训练次数 | 预计时间 |
|------|--------|--------|----------|----------|
| **简单** ⭐ | ETTh1, ETTh2 | 7 | 40 次 | ~30 分钟 |
| **中等** ⭐⭐ | ETTm1, ETTm2, exchange_rate, national_illness | 7-8 | 80 次 | ~2 小时 |
| **困难** ⭐⭐⭐ | weather, electricity, traffic, solar_AL | 21-862 | 80 次 | ~8 小时 |

---

## 使用方法

### 方法 1: 使用批处理脚本（推荐）

```batch
# 运行简单数据集（推荐先跑这个）
运行_简单.bat

# 运行中等难度数据集
运行_中等.bat

# 运行困难数据集（需要好电脑）
运行_困难.bat

# 运行全部（慎重！）
运行_全部.bat
```

### 方法 2: 命令行运行

```bash
cd repo
..\..venv\Scripts\python.exe run.py easy     # 简单
..\..venv\Scripts\python.exe run.py medium   # 中等
..\..venv\Scripts\python.exe run.py hard     # 困难
..\..venv\Scripts\python.exe run.py all      # 全部
```

---

## 给学长学姐的建议

### 快速验证（推荐）
先跑 `运行_简单.bat`，包含 ETTh1 和 ETTh2：
- ✅ 变量少（7 维），速度快
- ✅ 数据量小，训练快
- ✅ 结果与论文对比明显
- ⏱️ 总时间约 30 分钟

### 完整复现
如果简单数据集结果正常，再跑 `运行_中等.bat`

### 给学妹跑的
如果电脑配置一般，只跑 `运行_简单.bat` 就够了！
如果想发论文，建议跑完中等难度。

---

## 结果查看

运行完成后，查看结果：
```bash
..\..venv\Scripts\python.exe collect_results.py
```

生成的 `实验结果汇总.md` 包含：
- 每个数据集的 MSE/MAE
- 与论文结果的对比
- 差异分析

---

## 常见问题

### Q: 为什么 Traffic 要跑 50 个 epoch？
A: 论文规定 Traffic 数据集需要 50 个 epoch 才能收敛，其他数据集 30 个就够了。

### Q: 跑一半可以停止吗？
A: 可以，已训练的结果会保存在 `LongTermTSF_SEMixer/` 目录。

### Q: GPU 内存不够怎么办？
A: 只跑简单数据集，或者把 batch_size 减半。