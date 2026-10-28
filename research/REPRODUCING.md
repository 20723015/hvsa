# 运行与复现说明

所有命令从仓库根目录执行。建议使用独立环境；本次记录的训练解释器为Python 3.9.15，NumPy 1.26.4、PyTorch 2.7.0+cu118。依赖清单在[`requirements-review.txt`](../requirements-review.txt)，其中torch/torchvision不固定CUDA平台后缀，因安装平台而异。LLM/文献检索工具不是核心实验依赖。

## 先检查目录，不运行实验

```bash
python tools/research/catalog.py --check
python tools/research/prepare_inventory.py --check
```

第一条检查提交清单、SHA-256、Python/JSON语法及审阅导航链接；不导入协议，不启动训练。第二条检查随代码提供的历史冻结清单和20项源码摘要。源文件字节与换行在`.gitattributes`中保留。

## 可独立启动的小型示例

以下命令会实际创建新实验产物。本次整合没有执行这些密码/训练实验，避免干扰正在运行的计时任务。

```bash
# 点一：小规模真实链路正确性，只测HVSA。
python -m experiments.point1_compare.run_compare e1 --smoke --schemes hvsa

# 点一：正式GPVO/VLMRS实例化的小型三方案试跑。
python -m experiments.point1_compare.run_decided_comparison --pilot

# 点一：独立生成一轮新的MNIST安全训练数据；需要下载数据集。
python -m experiments.point1_compare.run_training --seeds 42 --rounds 1

# 点二：仅列出主矩阵，不启动训练。
python run_thesis_core.py plan
python run_plain_bench.py plan

# 点二：小型半异步验证链路，输出到独立目录。
python main_point2.py --dataset MNIST --model Tiny --seed 42 --deterministic --max-checkpoints 2 --max-ticks 30 --verify --vcm --output-dir results_review_point2 --run-id review_smoke
```

`--pilot`使用的是参数受限的小试，不是正式生产参数扫描。点二这里的`--vcm`是既有版本对齐，不是10月12日的新checkpoint更新矩校准。

## 冻结清单与正式单点

`research/resources/original_cost_inventory_20261020.json`是原历史batch清单的逐字节副本，SHA-256为`cce46db69060fb4656d6e50e58349addb9254dd0ad8b49a56b2efddb6dc2d696`。它只提供冻结源码清单，不代表已提供完整原始实验数据。

```bash
# 在新checkout准备正式contract期待的清单路径。
# 已有文件只校验、不覆盖；不会改写真实实验记录。
python tools/research/prepare_inventory.py --prepare

# 独立环境中运行新9次协议小试；不进入正式论文分母。
python -m experiments.point1_compare.run_full_scale_point --smoke
```

部分历史代码会优先寻找`tmp/point1_pairing_deps/`；正常安装`py-ecc==8.0.0`后仍可通过标准Python导入。审阅版不携带本机的临时依赖目录。

## 需要原始归档的命令

下列流程本来就引用不可变历史实跑，下载代码后不能凭一份清单代替这些证据：

- `run_full_scale_batch.py`复用三个原始成本点，须取得其`manifest/raw/messages/correctness/summary/sources`等完整归档；历史batch中的绝对路径还需保持可解析。单点新运行与历史批次验收是不同用途。
- `run_training_controls.py`复算已有60轮HVSA训练，依赖`training_control_contract.py`的固定来源、快照与源码归档。
- `validate_*.py`、`report_*.py`和多数`tests/test_point1_*artifacts.py`等需要对应的完整run目录。
- `tests/test_point1_full_scale.py`需要`POINT1_FULL_SCALE_TEST_RUN`指向一个已完成的真实点，并需要原三个复用点。其他`*_TEST_RUN`变量以各测试文件为准。

例如取得完整真实归档后才可运行：

```bash
python -m experiments.point1_compare.validate_full_scale path/to/completed_run
python -m pytest tests/test_point1_full_scale.py tests/test_point1_windows_artifact_io.py
```

正式run归档常包含用户向量、完整消息计量与源码副本，体积较大，采用本地单独交付。代码审阅分支不含这些原始数据，也没有上传模型、数据集或私钥。运行中的最后一点及两个中断前缀均不改写。
