# 数据格式与信息边界

## 正常源文件

一个NPZ文件，至少提供：

|字段|形状|定义|
|---|---|---|
|z_pu|N×28|带噪正常量测，P1..P14,Q1..Q14，100MVA基值标幺|
|state_true_pu_rad|N×27|真实Vm1..Vm14，theta2..theta14，角度rad|
|normal_estimated_state|N×27|正常量测的AC-WLS估计，仅供攻击者构造FDIA|

前两项供训练；第三项供攻击构造，不能用真实标签冒充AC-WLS结果。所有状态参考角theta1固定0。源数据不得包含攻击；`attack_label`若提供必须全0。数据按时间排序且不打乱，不同分区之间时间先后关系由配置边界保证。

真实数据未随代码发布。`prepare`只转换已有MATPOWER缓存，源工况的1%负荷扰动与后续sigma0.001pu传感器噪声不是同一件事；不要在转换时重复添加噪声。

## 生成攻击目录

`normal.npz`包含原测试段的正常对照及之前23步的正常上下文。`cases/*.npz`提供相同测试时间轴的9个场景：

- `z_pu`：实际受到篡改的连续量测，是在线输入；正常行不变。
- `attack_label`：当前时刻是否注入攻击，不根据历史窗口是否含攻击重新标注。
- `event_id`：0为正常，正整数标识完整攻击段。
- `target_node_mask`、`state_ratio`：构造审计，不是在线输入。
- `state_true_pu_rad`：真实状态监督/离线评价标签，攻击后保持不变。
- `original_index`：0起原始时间索引。

重放缓存记录攻击者截获的注入前合法数据；当15步攻击超过12步重放延迟时，最后几步仍读取过去合法原始数据，而不是递归使用已伪造量测。尺度场景共用重放的目标节点，减少位置差异的干扰。

`manifest.json`记录配置、SHA256和数量。评估器会校验文件哈希，不使用正常对照替换攻击期间的输入历史。正常对照只计算离线诊断误报，未参与攻击结果的阈值调整。

## 工件目录

`preprocessing.npz`包含`input_mean/input_scale/state_mean/state_scale`。`.pt`文件仅含family、seed、config和state_dict。`thresholds.json`保存每权重的系统级威布尔参数与数值阈值。`provenance.json`绑定标准化文件的哈希。改变任何一项后，应重新进行正常数据校准，不应继续声称阈值沿用。
