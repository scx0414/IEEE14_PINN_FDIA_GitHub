# IEEE14 PINN State Estimation and FDIA Detection

IEEE14交流静态状态估计与攻击检测研究代码。包含LSTM-PINN、P-sLSTM-PINN和xLSTM-Mixer-PINN，使用AC量测重构残差及系统级威布尔阈值，检测重放、尺度篡改和AC物理一致FDIA。

这是研究实验代码，不是部署级电网安全系统。P-sLSTM/Mixer是状态估计任务适配，不等同于原论文的预测模型。主线输入含当前时刻量测，是**状态估计**，不是未来状态预测。

## 安装和快速验证

Python3.10及以上；本地验证环境为Python3.12、PyTorch2.9、NumPy1.26。CPU可运行；正式实验建议CUDA。sLSTM使用保留完整记忆机制的vanilla后端，不需要编译CUDA扩展。

```bash
python -m pip install -e ".[dev]"
python -m ieee14_fdia --help
python -m pytest -q
python -m ieee14_fdia demo --network examples/ieee14_network.json --config configs/models.json --output runs/demo --device cpu --epochs 2
```

若需要与本地验证使用相同版本，可在安装命令中加`-c requirements-tested.txt`；Torch的CPU/CUDA发行版本按运行设备选择。

demo生成合成数据、训练小轮数LSTM、用正常段校准阈值并运行三类攻击，检查流程可执行。**demo的模拟估计状态不是AC-WLS解，其指标不属于研究结果。**

## 模型与检测流程

```text
当前/历史P、Q [B,L,28]
       ↓ LSTM / P-sLSTM / xLSTM-Mixer
估计状态 [B,27]：14 Vm(pu)＋13 theta(rad)，theta1=0
       ↓ 完整AC量测方程 h(x)
重构P、Q [B,28]
       ↓ 与收到的量测比较
节点误差 s_i=(P_i-P_hat_i)^2+(Q_i-Q_hat_i)^2
       ↓ 系统统计量 S=max_i(s_i)
S超过正常数据拟合的威布尔阈值 → 系统报警
```

- LSTM/P-sLSTM默认12步，Mixer默认24步；全部27维状态由模型输出。
- PINN训练损失：监督状态损失与AC量测一致性损失的加权和；各物理量组使用原实验的batch标准差归一化。
- 威布尔位置参数固定0，目标系统虚警率α=0.05：`tau=scale*(-log(alpha))**(1/shape)`。
- **目标5%不是实际误报率保证**。拟合偏差、分布变化及攻击后历史污染都可能增加实际误报。

## 数据准备

真实数据和检查点不随仓库发布。数据格式见[DATA_FORMAT.md](docs/DATA_FORMAT.md)。如果已有本项目的正常MATPOWER缓存，可显式转换：

```bash
python -m ieee14_fdia prepare --cache local/normal_cache.mat --output data/normal.npz
```

这里复用已计算的潮流结果，不会重新进行MATPOWER仿真。也可自行提供符合格式的正常NPZ。示例导纳矩阵见[examples/ieee14_network.json](examples/ieee14_network.json)，由本实验的IEEE14网络矩阵导出。

## 路线A：使用已有权重与冻结阈值

将命令中的`local/...`替换为你的文件位置。此导入只整理已有张量、标准化参数与阈值，不训练或重新拟合。

```bash
python -m ieee14_fdia import-legacy --checkpoints local/old_checkpoints --preprocessing local/preprocessing.npz --thresholds local/system5_normal_metrics.json --output artifacts/legacy
python -m ieee14_fdia generate-attacks --data data/normal.npz --network examples/ieee14_network.json --config configs/attacks_continuous15.json --output runs/tang15
python -m ieee14_fdia evaluate --attacks runs/tang15 --network examples/ieee14_network.json --artifacts artifacts/legacy --output runs/evaluation --device auto
```

旧权重文件名为`LSTM_2024.pt`、`P_sLSTM_2024.pt`、`xLSTM_Mixer_2024.pt`等，种子为2024/2025/2026；`*_parent.pt`不作为最终模型导入。必须配套旧标准化参数，不能改用新数据统计。

## 路线B：重新训练和正常校准

```bash
python -m ieee14_fdia train --data data/normal.npz --network examples/ieee14_network.json --config configs/models.json --split configs/split.json --family LSTM --seed 2024 --output artifacts/retrained --device auto
python -m ieee14_fdia calibrate --data data/normal.npz --network examples/ieee14_network.json --split configs/split.json --artifacts artifacts/retrained --alpha 0.05 --device auto
```

其余模型用`--family P_sLSTM`或`--family xLSTM_Mixer`；更换`--seed`运行多个种子。训练只读取训练/选模正常段，校准只读取后续正常段。`configs/split.json`使用原时间轴的半开边界；`train_end=16840`、`selection_end=18782`、`calibration_end=20724`。

这条路线是可移植eager训练，保留模型结构和物理损失，但不承诺逐位复现原CUDA Graph训练的优化轨迹；共享训练起点为原索引23，新训练划分也不同于最早旧权重使用的56%训练划分。仓库中的历史结果来自路线A，而不是这里重新训练的权重。

## 连续攻击设置

默认保持25908个原时刻的时间边界，测试为索引20724至25907：

|类型|配置|
|---|---|
|重放|随机3个非参考节点，P/Q替换为12步前的合法原始量测|
|尺度|同样的随机3节点，P/Q乘1.1或0.9，分别评估|
|AC相角FDIA|随机1/2/3节点，比例随机±10%～20%或±20%～30%，共6配置|

每段攻击15步，之后48步正常；每步5分钟。节点、方向和比例段内固定，下一段重抽。每场景有83段完整攻击，1245攻击点、3939正常点；最后剩3步正常。不同场景分别运行，不拼接成同一条时间线。**推理历史保留此前攻击，不根据真实攻击边界重置。**

借鉴唐论文的三类攻击以及DDET-MTD的AC比例攻击公式，但论文未明确的持续时间、幅度等属于本实验设置，不能称为严格复现。该生成器没有声称全部攻击通过BDD。

可选BDD检查需单独安装MATPOWER，见[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。Python训练/推理不依赖MATLAB。

## 已完成实验的参考结果

以下是整理前已完成的同一季度、旧模型三个种子的均值，不是合成demo或重新训练结果。每格为“检测率/实际系统误报率”。

|攻击|LSTM-PINN|P-sLSTM-PINN|Mixer-PINN|
|---|---:|---:|---:|
|重放12步|55.02% / 8.28%|64.34% / 7.60%|52.80% / 7.08%|
|尺度+10%|95.42% / 8.65%|96.89% / 8.28%|91.57% / 8.59%|
|尺度−10%|92.74% / 8.23%|96.31% / 8.50%|92.80% / 8.54%|
|FDIA6配置平均|100% / 19.28%|100% / 28.70%|100% / 49.56%|

逐种子与逐配置汇总见[results/reference](results/reference)。点级检测率与事件检出率分开计算；漏检事件不计入延迟中位数，但始终计为漏检。不同攻击比例下准确率不可直接比较。

## 目录

```text
src/ieee14_fdia/      模型、物理损失、攻击、阈值、训练与评估
src/.../_vendor/     xLSTM核心和其Apache2.0许可证
configs/             每模型参数、时间划分、攻击时序
examples/            IEEE14网络矩阵JSON
matlab/              可选盲AC-WLS/BDD检查
tests/               物理公式、阈值口径、连续历史及模型验证
docs/                数据格式、复现边界与上传说明
results/reference/   小型历史实验汇总CSV
```

数据、权重、预测数组、日志、论文和运行产物默认被Git忽略。发布前阅读[PUBLISHING.md](docs/PUBLISHING.md)，第三方代码说明见[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。项目原创部分尚未指定开源许可证，未默认替维护者选择。
