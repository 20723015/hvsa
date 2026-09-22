# 检索日志工作区

本目录为论文新颖性表述的证据基础。**引用检索结论前必须读 `检索日志.md` 第 3 节与第 6 节**。

## 文件说明

| 文件 | 内容 |
|---|---|
| `检索日志.md` | 正式检索日志：检索设置、检索式与命中清单、**覆盖度检验**、参考文献核验、最接近工作、**局限性声明** |
| `方向1_异步半异步FL.md` | 方向 1 完整报告（必读清单 24 条 + 最接近 3 条 + 检索式 + 局限） |
| `方向2_可验证加权聚合.md` | 方向 2 完整报告（必读清单 18 条 + 最接近 3 条 + 检索式 + 局限）——**创新点 2 新颖性的直接依据** |
| `方向3_分层可验证SA.md` | 方向 3 完整报告（必读清单 18 条 + 最接近 3 条 + 检索式 + 局限）——**创新点 1 的坐标系** |
| `refs_verified.bib` | `ref_point1.md` 36 条著录转 BibTeX，**逐条经 DOI/arXiv 反查**（28 Crossref + 6 arXiv + 2 未验证） |
| `refs_report.json` | 上述核验的逐条结果（含未验证标记） |
| `arxiv_hits.json` / `crossref_hits.json` | 两库检索的原始命中（含检索式与全部命中条目） |
| `agent_raw/` | 子智能体执行检索时的原始输出（查询脚本、API 原始 JSON、文本抽取），供复核 |

生成脚本（仓库根目录）：`lit_search.py`（arXiv）、`lit_search_crossref.py`（Crossref）、`build_refs_bib.py`（BibTeX 构建 + DOI 核验）、`build_search_log.py`（日志汇总）。

## 三条必须先知道的结论

### 1. 覆盖度检验显示查询式检索会漏检已知条目

拿**已知存在**的对照条目反查检索式：6 条中仅 GPVO-FL 被精确命中，**VerifyNet / VLMRS / MAFSA / DVAFL 全部漏检**，VeriFL 只命中衍生工作。

→ **不得以「某方向命中数低」论证新颖性。** 低命中更可能反映检索式与数据库覆盖的局限。

### 2. 「定长验证通信」不是全新性质（影响创新点 1 表述）

已有工作主张验证材料/通信与模型维度无关：**e-SeaFL（arXiv:2304.03841, 2023，最早）**、**Bottoni et al.（arXiv:2509.05162，"size of each authenticator is constant and does not depend on the column length d"）**、**Qin et al.（arXiv:2603.10692，"zero additional communication overhead"，最强主张）**。

→ 本文只能主张**在「1 云 + M 边 + N 端、边缘验证云端聚合、单 TA、PRG 掩码组包」这一组合下**实现该性质，**不能**主张该性质本身是新的。详见 `方向3_分层可验证SA.md` 开头。

### 3. 「验证决策/规则」已有先例（影响创新点 2 表述）

已有 4 条工作验证的是**决策/规则**而非仅算术结果：**Lotto（USENIX Sec'24，服务器不得选择参与者）**、**PRoVeFL（验证 Krum/TrimmedMean 等鲁棒规则被正确执行）**、**End-to-End Verifiable and Robust FL（NIZK 验证离群点排除）**、**Nautilus（ZKP 验证调度公平性）**。

→ 本文的差异点应收敛为：验证对象是**声明的权重值**（而非决策本身）；验证方式是**从公开量独立复算**（而非验证证明）；且有**不可复算策略 fail-closed 拒绝**语义。这三条在检索中**未发现对应物**。详见 `方向2_可验证加权聚合.md` 第 0 节。

## 待办（写作前）

1. **优先取得两篇全文**（IEEE 付费墙，本次取不到，直接决定创新点 1 表述边界）：
   - `10.1109/tmc.2026.3679423` — Fan et al., "Practical Multi-Key Secure Aggregation With **Homomorphic Hash Verification**"
   - `10.1109/icc59461.2026.11587871` — Liao et al., "Privacy-Preserving Verification of Secure Aggregation for **Hierarchical** Peer-to-Peer FL"
2. **补做引文图滚雪球**（前向引用检索）：至少覆盖 FedBuff、Papaya、FedAT、FedCompass、FedBASA 的施引文献。这是本次检索最大的方法论缺口。
3. **引用 VERSA / VOSA 前先读其安全质疑**：`10.1109/tdsc.2023.3253082`、`10.1109/tdsc.2023.3272338`、`10.1109/tdsc.2024.3352170`。
4. **不要引用** `arXiv:2605.24054`（作者已声明撤稿）。
5. 补读 SSRN 5018523 与 KBS 2026（`10.1016/j.knosys.2026.115310`）——标题与本文主题高度相关但摘要未取到。

## 表述纪律

正文如需保留「首次」类表述，措辞须限定为**「据本文检索所知（检索式与范围见附录）」**，且**不得**用于对 IEEE/ACM 期刊文献的排他性断言。检索未覆盖 IEEE Xplore / ACM DL / SpringerLink 全文，未做引文追踪，未覆盖中文文献。
