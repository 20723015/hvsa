# 研究点一：2026年文献补充与对比对象更新

检索日期：2026-10-17。范围：可验证安全聚合、云边协同、单掩码、轻量验证与动态参与。依据为本地原始论文、出版商页面及作者机构资料；检索结果不等于已完成所有候选的全文复现。

后续更新：已完成[第二轮近邻检索](POINT1_NEAREST_WORKS_SECOND_SEARCH_20261017.md)，新增OVP-FL、云边同态MAC方案、OPSA、TriVer与LPV-FL的核对结果。部分2025年工作在验证外包和边缘验证上比泛泛的2026年应用论文更直接，应按技术接近程度而非仅按年份选文献。

## 1. 建议

建议补充直接相关的2026年工作。最应补的是 SAEV-FL；VLMRS 已在当前会议稿中按2026年正式卷期引用，应保留。DSVFL 适合补充动态参与和验证方向；VCFedAdam、VDBFL 可按正文讨论范围选择，不要求全部新增实验。

文献引用、机制比较和完整性能复现是三个不同层次。新增引用应伴随具体讨论，不把所有检索命中的论文都列为性能基线，也不替换仍然最直接的 GPVO-FL。

## 2. 推荐清单

| 文献 | 出版信息 | 与研究点一的关系 | 本轮建议 | 核对程度 |
|---|---|---|---|---|
| SAEV-FL | IEEE TIV 11(1), 133–149, 2026；DOI 10.1109/TIV.2025.3599909 | 与 GPVO 作者相同；云/边/端/可信TA，单掩码、同态哈希、在线集合辅助 | 优先补引用及功能/机制对照；作为新增性能基线的首选评估对象 | 已核对本地原文摘要、系统/威胁模型、图3及算法2–4，并目视复核模型和完整流程；正式卷期核对作者机构资料 |
| VLMRS | IEEE TDSC 23(1), 403–416, 2026；2025-09-04首次在线；DOI 10.1109/TDSC.2025.3606413 | 双掩码、掉线恢复、多轮参数复用、客户端验证 | 已引用；继续作为原计划第二条完整性能基线 | 出版商元数据/摘要与本地原文已核对，完整复现待实施 |
| DSVFL | IEEE TMC 25(4), 4618–4631, 2026；2025-10-27首次在线；DOI 10.1109/TMC.2025.3625654 | 动态安全聚合与验证，覆盖掉线下的恢复/验证，并含梯度选择 | 建议补相关工作，比较动态参与、验证位置与恢复依赖；暂不强制完整性能复现 | 已核对本地正式PDF首页、摘要与威胁模型；未完成逐算法复现审查 |
| VCFedAdam | Sensors 26(18), 5814；2026-09-14发表；DOI 10.3390/s26185814 | 低维草图中聚合与验证、固定在线集合、承诺绑定验证，结合FedAdam | 可选补充2026年新方向；暂不加入主性能图 | 出版商日期、摘要及部分正文已核对，全文协议/证明未逐项审查 |
| VDBFL | Computer Communications 256, 108575；2026-07-01卷期；DOI 10.1016/j.comcom.2026.108575 | 去中心化、拜占庭鲁棒与可验证聚合 | 如正文讨论鲁棒性或可信边缘边界，可补引用；不列为首轮强制实验 | 出版商题录/摘要级，未取得并核对全文 |

时间区分：DOI中包含2025不代表正式卷期不能写2026；同时，2026卷期也不等于2026年首次公开。检索数据库的“抓取日期”不能替代发表日期。

当前 `E:/bs2026/main.tex:525–526` 已将 VLMRS 正确著录为 TDSC 23(1), 403–416 (2026)。无需为了增加2026年份再重复引用同一论文。

## 3. 最应加入的 SAEV-FL

### 3.1 为什么值得引用

该论文与研究点一的结构接近：用户、边缘、云和单个可信 TA；采用单掩码和同态哈希，TA依据实际在线集合提供辅助信息。因此，“云边协同+可信TA+单掩码”不能单独作为 HVSA 独有性质。

本地文件是IEEE作者接受稿，页眉仍有模板年份。正式引用应使用正式著录信息，不能把页眉“JUNE 2024”当作其发表年份。

### 3.2 应如何区别

| 维度 | SAEV-FL原文所述 | 当前HVSA研究点一 |
|---|---|---|
| 掩码结构 | CRT系数结合扰动的单掩码 | 客户端与TA共享种子派生的加性掩码 |
| 证明生成 | 客户端对本地向量作同态哈希并盲化，随更新上传 | 边缘对组掩码和作哈希并签署完整元数据 |
| 验证位置 | 客户端解密、计算聚合结果哈希并验证 | 边缘验证签名与哈希关系，重构共识模型 |
| 全局结果可见性 | 将全局参数保密作为明确设计目标，云分发聚合密文 | 允许云获得原始明文聚合和，全局/组聚合保密不是目标 |
| TA辅助信息 | 原文按在线集合计算辅助解密参数K与去盲因子R | 按在线集合计算总掩码及其哈希 |
| 客户端主要成本 | 掩码、证明生成、解密和结果验证 | 掩码与上传；完整验证在边缘执行 |

表格是机制区分，不是“HVSA全面更强”的判断。SAEV-FL的全局参数保密目标与HVSA不同，性能实验必须列出这一差别。

### 3.3 实验是否要补

先做完整消息/操作清单和 `N=20, M=5, d=7850` 的合法参数试跑，确认CRT域、掩码、解密及哈希验证一致，再决定是否加入完整扫描。

如实施，至少报告客户端掩码、证明、解密、验证四项，以及TA和全网通信；不得删掉SAEV-FL原文的全局参数保护操作后仍以原方案名称出图。若只做其原语微基准，明确标注范围，不能称为完整协议复现。

第一轮完整性能比较仍按“HVSA/GPVO优先、VLMRS随后”推进；SAEV-FL的引用和机制表先补，不以其完整复现阻塞首轮结果。完成条件评估后，若增加一条新性能基线，优先SAEV-FL而非仅因年份更新就选择任意2026论文。

## 4. DSVFL、VCFedAdam、VDBFL怎么使用

- **DSVFL**：用于说明2026年正式发表的工作已联合研究聚合隐私、验证和客户端掉线。其梯度选择属于额外训练机制；若未来比较准确率或时间，需要先统一训练任务并解释该差异。本轮不依据摘要中的百分比直接与HVSA实验相减。
- **VCFedAdam**：用于补充2026年9月的新方向，其压缩比例、草图重构与FedAdam会改变通信和训练行为。只宜先作相关工作或另设压缩实验，不把其论文报告的压缩收益当作同维度安全协议收益。
- **VDBFL**：用于说明去中心化和拜占庭鲁棒验证的扩展方向。HVSA当前以半诚实客户端和诚实验证边缘为边界，不能声称同等鲁棒性；摘要级材料不支持精确的协议成本和合谋阈值比较。

## 5. 可加入相关工作的文字

建议在单掩码/分层安全聚合与验证工作段落中具体讨论，而不是只增加参考文献条目：

> Zhang等提出SAEV-FL，在云边协同架构下结合中国剩余定理和扰动技术实现单掩码聚合，并由客户端完成聚合结果的解密与验证，同时考虑全局参数保密。Li等提出VLMRS，通过双掩码、掉线恢复和参数复用支持多轮可验证聚合。另有DSVFL将动态参与下的安全聚合与验证结合研究。本文关注允许发布全局聚合结果的单可信TA场景，将组证据生成与结果验证置于边缘，重点评估客户端负担以及边缘和TA所承担的成本。

如正文涉及压缩或自适应优化，可再加：

> 近期VCFedAdam进一步在低维草图空间中结合安全聚合、承诺绑定验证与FedAdam，以降低传输及计算成本。其压缩和优化目标与本文保持原维度聚合结果的研究设置不同。

以上是供后续编辑的草稿，不是已写入会议稿或学位论文的内容。关于方案优劣的量化结论必须等公平实验完成后填写。

## 6. 新增著录草稿

SAEV-FL正式卷期由作者机构公开发表清单交叉确认；DSVFL取本地正式PDF首页；另外两篇取出版商页面。实际加入前按论文模板统一姓名与期刊缩写。

```bibtex
@article{SAEVFL2026,
  author  = {Zhang, Shiwen and Ren, Feixiang and Liang, Wei and Li, Kuanching and Ling, Nam},
  title   = {{SAEV-FL}: Lightweight Secure Aggregation and Efficient Verification Scheme for Federated Learning in Cloud-Edge Collaborative Environment},
  journal = {IEEE Transactions on Intelligent Vehicles},
  year    = {2026},
  volume  = {11},
  number  = {1},
  pages   = {133--149},
  doi     = {10.1109/TIV.2025.3599909}
}

@article{DSVFL2026,
  author  = {Li, Mengqian and Tian, Youliang and Zhang, Junpeng and Xiong, Jinbo and Cui, Xinhua and Ma, Jianfeng},
  title   = {Enhancing Federated Learning in Edge Computing With Secure Aggregation and Dynamic Verification},
  journal = {IEEE Transactions on Mobile Computing},
  year    = {2026},
  volume  = {25},
  number  = {4},
  pages   = {4618--4631},
  doi     = {10.1109/TMC.2025.3625654}
}

@article{VCFedAdam2026,
  author  = {Si, Zhiwei and Liao, Xiuheng and Wu, Ziang and Li, Tianhui and Su, Chunhua},
  title   = {{VCFedAdam}: A Verifiable and Compressed {FedAdam} Aggregation Scheme for Federated Learning in Distributed Sensor Networks},
  journal = {Sensors},
  year    = {2026},
  volume  = {26},
  number  = {18},
  pages   = {5814},
  doi     = {10.3390/s26185814}
}

@article{VDBFL2026,
  author  = {Liu, Dengzhi and Li, Qingyuan and Shen, Jun and Tan, Haowen},
  title   = {{VDBFL}: Verifiable Privacy-Preserving Federated Learning With Decentralized Byzantine-Robust Aggregation in Edge Computing},
  journal = {Computer Communications},
  year    = {2026},
  volume  = {256},
  pages   = {108575},
  doi     = {10.1016/j.comcom.2026.108575}
}
```

## 7. 暂不直接纳入的检索线索

检索还返回SVTAgg、FedAggVerify等2026年题录，但本轮未取得足够的出版商/作者原始材料核验，暂不依据聚合网站的摘要下安全性结论，也不直接写入新增正式著录。

VCRFL的出版商索引列为FGCS 184（November 2026），已可检索到摘要，但当前日期为2026-10-17，本轮未确认其首次在线日期。不将其简单描述为“已于2026年11月发表”；取得正式在线日期和全文后再决定是否加入。

这是针对本次问题的定向检索，不是2026年文献的穷尽调查。当前未完成的全文核对事项保留为待办。

## 8. 原始来源

- [SAEV-FL DOI](https://doi.org/10.1109/TIV.2025.3599909)；[作者Nam Ling机构公开发表清单](https://www.scu.edu/media/school-of-engineering/pdfs/faculty-and-staff/faculty-pages/Ling-CV.pdf)。
- SAEV-FL本地原文：`E:/Essay/DFL/SAEV-FL_Lightweight_Secure_Aggregation_and_Efficient_Verification_Scheme_for_Federated_Learning_in_Cloud-Edge_Collaborative_Environment.pdf`，重点为PDF第1、2、5、7、8页。
- [VLMRS出版商记录](https://doi.org/10.1109/TDSC.2025.3606413)。
- [DSVFL DOI](https://doi.org/10.1109/TMC.2025.3625654)；本地正式PDF：`E:/Essay/DFL/Enhancing_Federated_Learning_in_Edge_Computing_With_Secure_Aggregation_and_Dynamic_Verification.pdf`，重点为第1、2页。
- [VCFedAdam出版商全文与日期](https://www.mdpi.com/1424-8220/26/18/5814)。
- [VDBFL出版商摘要](https://www.sciencedirect.com/science/article/abs/pii/S0140366426001659)。
- [VCRFL出版商索引](https://www.sciencedirect.com/science/article/pii/S0167739X26002694)，仅作进一步核对线索。
