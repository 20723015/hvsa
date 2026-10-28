# Paper Survey & Integration Analysis

> 检索日期: 2026-07-31
> 项目: Hierarchical Verifiable Secure Aggregation (HVSA) for Federated Learning

---

## 模块级分析

### 1. PRG Masking → Dropout Resilience

**当前状态:** X25519 DH + HKDF-SHA256 种子协商，在线用户列表处理掉线。

**相关论文:** LightSecAgg (USENIX Security 2026) — 基于 Lagrange 插值的掩码求和技术，支持分层网络中的弹性掉线。

**为什么可缝:** 当前掉线处理方式简单（在线用户列表排除种子），但 Point 2 半异步路径需要更鲁棒的掉线恢复。LightSecAgg 的 Lagrange 方法允许掩码恢复而无需重新运行协议轮次。

**如何缝:** 替换 `ta.py` 中 `trigger_global_mask_computation()` 的掩码聚合逻辑，保留 X25519 密钥协商，将掩码线性组合替换为 Lagrange 系数。

**难点:** 中 | **影响:** 更好的掉线处理 | **改动范围:** ta.py 核心函数

---

### 2. LHH Verification → Efficient Batch Verification

**当前状态:** 基于 NIST P-256 的 LHH，每维度一个生成器，O(1) 验证开销。

**相关论文:** Efficient Multi-Client Verifiable Secure Aggregation (ACM TOPS 2025) — 使用 Lagrange 多项式承诺和批量验证。

**为什么可缝:** 当前 LHH 每维度需要一个生成器，大模型设置成本 O(d)。多项式承诺方法将设置成本降至 O(log d) 或 O(1)，同时保持可验证性。

**如何缝:** 在 `utils.py` 中将 `LHH` 类替换为多项式承诺方案，接口不变（`LHH.compute()` / `LHH.verify()`）。

**难点:** 中 | **影响:** 大模型设置更快，证明更小 | **改动范围:** utils.py 原语替换

---

### 3. Gradient Compression + Secure Aggregation

**当前状态:** 无梯度压缩，客户端上传完整模型。

**相关论文:** GCCL: Gradient Compression Compatible with Secure Aggregation (IEEE INFOCOM 2025) — 使 top-k 稀疏化与安全掩码兼容。

**为什么可缝:** 最大的性能瓶颈。客户端→Edge 通信是主要带宽瓶颈。GCCL 使用共享种子进行索引选择 + 两阶段协议，使 top-k 稀疏化与 PRG 掩码兼容。

**如何缝:** 在 `utils.py` 添加 `compress_and_mask()`，在 `user.py` 的 `mask_and_upload_model()` 中调用。Edge 添加 `decompress_pre_aggregate()`。压缩后模型意味着更小的 LHH 哈希。

**难点:** 高 | **影响:** 通信减少 70%，准确率 < 1% 下降 | **改动范围:** user.py + edge.py + utils.py

---

### 4. Semi-Async VCM → Adaptive Compensation

**当前状态:** Direct-VCM 补偿 `C_k(t) = S_k · γ_vcm · (W(t) - W(t_k))`，固定 γ_vcm 和指数衰减 ρ_i。

**相关论文:** Semi-Asynchronous FL with Staleness-Aware Checkpointing (2025) — 动态阈值和自适应惩罚变体。

**为什么可缝:** 固定 γ_vcm 无法适应模型变化速率。自适应方法根据实际梯度变化调整——模型快速变化时增加惩罚，稳定时降低惩罚。

**如何缝:** 在 `main_point2.py` 中将固定 γ_vcm 替换为跟踪 `W(t) - W(t-1)` 范数的自适应函数。`checkpoint_trigger()` 逻辑不变。

**难点:** 低 | **影响:** 收敛更稳定，高掉线场景更鲁棒 | **改动范围:** main_point2.py 3-5 行

---

### 5. LLM Agent Orchestration (In Progress)

**当前状态:** 已有 `LLMReasoner`、`RuleEngine`、`OrchestrationAgent` 雏形。

**相关论文:** DAWN (AAAI-26) — Parametric Resonance (SVD 去噪 + 子空间对齐) + Structural Gravity (FGW 距离正则化)。

**为什么可缝:** DAWN 的 Parametric Resonance 解决异构客户端更新聚合问题，与 OrchestrationAgent 做客户端选择时面临的问题相同。Structural Gravity 可指导 RuleEngine 进行全局一致的调度决策。

**如何缝:** 在 `orchestrate_agent.py` 中将 Parametric Resonance 添加为可选聚合策略。RuleEngine 保持安全约束，Structural Gravity 作为软约束引导 agent 建议。

**难点:** 进行中 | **影响:** 更智能的编排，安全边界不变 | **改动范围:** orchestrate_agent.py + rule_engine.py

---

### 6. Byzantine Robustness (Optional)

**当前状态:** 假设用户半诚实。

**相关论文:** Verifiable Aggregation for FL with Byzantine Robustness (2025) — 阈值签名 + Merkle 树证明。

**为什么可缝:** 如需加强威胁模型到恶意用户，此论文将拜占庭鲁棒性与可验证性结合。阈值签名与 Ed25519 设置兼容。

**如何缝:** 在 `cloud.py` 聚合流程添加拜占庭滤波器，在 `edge.py` 验证添加 Merkle 树证明。需仔细设计，对威胁模型有重大影响。

**难点:** 高 | **影响:** 更强安全保证，复杂性增加 | **改动范围:** cloud.py + edge.py + utils.py

---

## 优先级排序

| 优先级 | 模块 | 难度 | 影响 | 改动范围 |
|--------|------|------|------|----------|
| P0 | 自适应 VCM (4) | 低 | 收敛稳定性 | 3-5 行 |
| P0 | Agent 编排 (5) | 进行中 | 自动化 | 已在做 |
| P1 | 梯度压缩 GCCL (3) | 高 | 70% 通信减少 | user.py + edge.py |
| P1 | 多项式承诺 LHH (2) | 中 | 大模型设置更快 | utils.py 原语替换 |
| P2 | LightSecAgg 掉线弹性 (1) | 中 | 更好的掉线处理 | ta.py 核心函数 |
| P3 | 拜占庭鲁棒性 (6) | 高 | 更强的威胁模型 | 仅当需要时 |

## 建议

1. 先完成 P0（自适应 VCM + Agent 编排）——基本已在做
2. 再做 P1（GCCL 梯度压缩）——当前项目最大空白
3. P2 和 P3 锦上添花，可在论文写作后添加