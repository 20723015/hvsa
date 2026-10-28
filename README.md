# HVSA：分层联邦学习可验证安全聚合

本仓库包含两个研究点的协议实现、实验入口、配置、测试及设计材料。面向代码审阅的分类入口在 [research/](research/README.md)。源码保留原路径，方便对照已冻结实验的源码摘要。

| 阅读目的 | 入口 | 主要源码 |
| --- | --- | --- |
| 研究点一：分层可验证安全聚合及跨方案比较 | [实验分类](research/point1/README.md) | `experiments/point1_compare/` |
| 研究点二：半异步 checkpoint、版本补偿与提交验证 | [实验分类](research/point2/README.md) | `main_point2.py`、`commit_tx.py` |
| 客户端、边缘、云、TA与密码组件 | [公共代码](research/shared/README.md) | `user.py`、`edge.py`、`cloud.py`、`ta.py`、`utils.py` |
| 早期探索、旧基准和辅助脚本 | [历史与辅助代码](research/archive/README.md) | `examples/`、`baselines/`及旧批跑脚本 |
| 完整文件清单与所属实验 | [机器可读目录](research/catalog.json) | `tools/research/catalog.py`生成 |
| 安装、运行及原始归档要求 | [复现说明](research/REPRODUCING.md) | `requirements-review.txt` |

建议师姐先看两个研究点的分类表，再按每行的“入口 → 配置/口径 → 核验/测试”阅读。各表区分已实现、历史探索和仅完成设计的内容。

## 当前范围

- 研究点一包含E1正确性、E2计算/通信、E3多轮、E4掉线、E5异常、E6消融、E7真实训练及可信假设对比。2026-10-28审阅快照时，完整跨方案规模扫描尚有最后一个工作点在运行；代码齐备不等于全部结果已验收。
- 研究点二现有代码支持半异步checkpoint、版本对齐、策略/结果核验和联合提交。2026-10-12的checkpoint更新矩校准方案是独立的设计稿，尚未接入完整安全训练，不能用既有VCM或其他moment实验替代其效果验证。
- GPVO/VLMRS为本项目明确说明选择的实例化，参数、可信设施和后端差异见[实例化决策](docs/POINT1_BASELINE_DECISIONS_20261019.md)。本仓库不提供原作者实现等价或SOTA认证。

## 最小检查

```bash
python -m pip install -r requirements-review.txt
python tools/research/catalog.py --check
python tools/research/prepare_inventory.py --check
```

安装与实验命令供独立环境使用。正在进行正式计时实验的同一台机器上，不要同时安装环境、运行训练或启动另一套密码基准。

审阅分支提交源码、配置、测试、说明和一份8KB历史冻结清单。数据集、原始实验归档、图表、模型与进程中的私钥状态仍保留在原本的本地实验目录；下载代码不会自动获得这些原始证据。正式归档核验所需文件见[复现说明](research/REPRODUCING.md)。

审阅分支采用独立快照历史，便于直接分享；它不承接主分支的旧历史。一个含本地API凭据的旧Agent开销示例已排除，详细范围见[历史代码说明](research/archive/README.md)。
