# 研究点一：独立明文FedAvg训练对照

日期：2026-10-26。新增两个实际闭环：同编码明文与独立浮点FedAvg，种子42/43/44各20轮，共60组配对、120个分支训练回合。同编码分支60/60逐轮重现旧HVSA的基模型、20个量化本地更新、整数和及测试正确数；旧来源的60次完整安全聚合和300次边缘验证不重复计作新执行。

## 设置与对照含义

MNIST训练60000/测试10000，Linear(784,10)、7850参数、20客户端IID等量划分、全员参与，本地SGD学习率0.01、1epoch、batch64，CPU单线程。两分支共享初始参数、划分和batch随机种子，后续各自从自己的上轮模型训练。浮点分支实际重新训练本地模型，不复用量化分支更新。

量化为float32乘10^6后截断int64，精确整数和除10^6再除20，再转float32写回。浮点对照将自己的float32本地模型转float64求均值，再转float32写回。

## 末轮结果

|种子|同编码明文/HVSA准确率%|独立浮点准确率%|浮点减同编码/百分点|参数最大坐标差|预测分歧/10000|
|---|---:|---:|---:|---:|---:|
|42|86.92|86.92|+0.00|1.8262072e-05|0|
|43|86.71|86.71|+0.00|1.8527731e-05|0|
|44|86.78|86.78|+0.00|1.8515857e-05|0|

末轮同编码均值86.803%，样本标准差0.107个百分点；浮点均值86.803%，样本标准差0.107个百分点。全部配对轮的最大参数坐标差为1.85277313e-05，最大预测分歧为1/10000。三种子的全部轨迹保留。

![逐种子训练准确率](../figures/point1_compare/20261026_training_controls/training_accuracy.png)

![逐轮配对差值](../figures/point1_compare/20261026_training_controls/training_differences.png)

## 核验与解释

独立核验重新计算全部60组聚合、两分支连续性及120组测试预测；按预声明规则每轮每分支选客户端(seed+round)%20，另外重新执行120次本地SGD并逐参数核对。它是本地SGD抽样复算，不表示离线重跑了全部2400个客户端训练。驱动执行了全部客户端训练，实际本地模型及随机数均归档。

全部60组配对的准确率差值（浮点减同编码）范围为-0.01至+0.01个百分点；有限种子不能证明统计等价或精度领先。HVSA与同编码明文的逐轮一致性和独立浮点分支的微小量化差异分别解释。未新增GPVO/VLMRS训练曲线，未更改已录用论文/PPT，不提供训练速度排名。

## 来源

- 新运行：`results/point1_compare/20261026-041744-080291_training_controls`，manifest SHA-256 `0ed1b6aed91ecacb89856f12725717a7ca196ec7bb4eac15f27e14acda5c2e6a`。
- 旧安全聚合来源：`results/point1_compare/20261018-203010-459621_e7_hvsa_v2`，manifest SHA-256 `b50a882c9388287f709c14b660112d93d61b501037c622fc7071cba27bff978a`，归档字节保持不变。
- [运行前规格](POINT1_TRAINING_CONTROL_STAGE_20261026.md)。
- [分析数据](../figures/point1_compare/20261026_training_controls/analysis.json)、[分析与图摘要](../figures/point1_compare/20261026_training_controls/analysis_manifest.json)。

两张最终PNG已实际目视核查，未见标题/图例/坐标重叠或裁切；SVG为同图导出，未单独渲染。[视觉核查凭据](../figures/point1_compare/20261026_training_controls/visual_review.json)。

[正式10项检查](POINT1_TRAINING_CONTROL_TEST_RECEIPT_20261026.json)、[证据凭据](POINT1_TRAINING_CONTROL_EVIDENCE_AUDIT_20261026.json)。
