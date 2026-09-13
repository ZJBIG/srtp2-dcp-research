# DCP / AIStation 训练总指南

本仓库当前的正式目标是：在 **Original DCP** 基础上训练并验证一条已冻结的主线：

```text
Original DCP frozen representation
        ↓
relative importance predictor
        ↓
availability-aware bounded Prompt Adapter
        ↓
direct-task continuous UtilityGate
        ↓
frozen DCP classifier / MM-IMDb BCE
```

工程侧已经完成路径迁移、依赖入口和预检脚本；研究侧仍需在 AIStation 上完成正式训练、消融和多 seed 验证。**当前不能宣称性能提升已经完成**。


## 1. 一句话结论

- 正式训练入口只有一个：`bash run_aistation.sh`
- 正式环境优先选择：Python 3.8 + GPU PyTorch + CUDA 可用
- 正式协议已经锁定：relative importance + bounded Adapter + direct-task continuous Gate + 20 epoch + seed 0
- 首次启动前必须先跑：`bash aistation_preflight.sh`
- 首次正式训练前必须先跑 2-batch 跨阶段检查


## 2. 平台内必需目录

本地已生成可直接上传的包：`aistation_upload/` 与 `aistation_upload.zip`；ZIP 包内以 `code/`、`data/` 等目录为根，解压后应得到下述结构。

假设所有文件已上传并解压到同一个工作目录，推荐结构如下：

```text
workspace/
├── code/
│   └── Deep_Correlated_Prompting-main/
│       ├── run.py
│       ├── run_aistation.sh
│       ├── setup_aistation_env.sh
│       ├── aistation_preflight.sh
│       ├── requirements_aistation.txt
│       ├── clip/
│       ├── tools/
│       └── tests/
├── data/
│   └── mmimdb/
│       ├── arrow/
│       │   ├── mmimdb_train.arrow
│       │   ├── mmimdb_dev.arrow
│       │   └── mmimdb_test.arrow
│       └── missing_tables/
│           ├── mmimdb_train_missing_both_07.pt
│           ├── mmimdb_dev_missing_both_07.pt
│           └── mmimdb_test_missing_both_07.pt
├── cache/
│   └── clip/
│       └── ViT-B-16.pt
├── experiments/
│   └── dcp_mmimdb_reproduction/
│       └── checkpoints/
│           └── dcp_mmimdb_reproduction_seed0_seed0/
│               └── version_2/
│                   └── checkpoints/
│                       └── epoch=3-step=507.ckpt
└── outputs/
    └── direct_task_full_logs/
```

如果平台实际挂载路径与上面不同，不要改源码，只在 AIStation 任务页或 Shell 中注入环境变量。


## 3. 前期检查清单

### 3.1 文件与数据

在“数据管理 > 文件管理”中确认：

1. `code/Deep_Correlated_Prompting-main` 存在，且包含 `run_aistation.sh`。
2. 三个 Arrow 文件存在且非空。
3. 三个 missing table 存在且非空。
4. `ViT-B-16.pt` 存在且非空。
5. Original DCP checkpoint 存在且非空。
6. `outputs/direct_task_full_logs` 目录可写。

预检脚本会强制校验以下行数与字段：

| 文件 | 行数 | 必需字段 |
| --- | ---: | --- |
| `mmimdb_train.arrow` | 15552 | `image`, `plots`, `label`, `genres`, `image_id`, `split` |
| `mmimdb_dev.arrow` | 2608 | 同上 |
| `mmimdb_test.arrow` | 7799 | 同上 |

### 3.2 是否需要创建“数据集”

首次运行建议直接使用“文件管理”中的目录挂载，因为代码、数据、权重和输出可以保持同一相对结构。

如果管理员强制使用“数据集管理”，流程是：

1. 进入“数据管理 > 数据集管理”。
2. 单击“创建数据集”。
3. 名称填写 `dcp_mmimdb_seed0`。
4. 导入路径选择 MM-IMDb 数据目录。
5. 数据类型选择“其他”。
6. 创建 V001 后发布为“个人”。
7. 在训练任务中选择该版本，并通过环境变量注入实际挂载路径。

注意：数据集管理一次只能选择一个版本，且仅适用于共享文件存储（NFS、GPFS 等），对象存储不支持。

## 4. 创建开发环境

进入：

```text
AI 开发 > 模型开发 > 开发环境 > 创建
```

推荐配置：

| 字段 | 建议值 |
| --- | --- |
| 名称 | `dcp_dev_seed0` |
| 框架 | PyTorch |
| 镜像 | GPU 版，Python 3.8，优先 PyTorch 2.0.x + CUDA 11.8 |
| 部署 | 单机 |
| GPU | 1 张 |
| CPU | 8 核起步 |
| 内存 | 32GB 起步 |
| Shm | 8GB 或 16GB |
| 数据 | 选择工程根目录，使用“直接使用” |
| 启动命令 | 保持镜像默认 |

已验证组合参考：

```text
Python 3.8.20
PyTorch 2.0.1+cu118
torchvision 0.15.2+cu118
```

环境创建成功后，单击环境名称进入 Shell 或 Jupyter。


## 5. 定位源码并验证 GPU

在开发环境 Shell 中执行：

```bash
nvidia-smi
```

如果 Shell 默认不在源码目录，先定位：

```bash
find $HOME -type f -name run_aistation.sh -print
```

进入脚本所在目录：

```bash
cd <run_aistation.sh 所在目录>
pwd
ls -lah
```

应看到：

```text
run.py
run_aistation.sh
setup_aistation_env.sh
aistation_preflight.sh
requirements_aistation.txt
clip/
tools/
tests/
```

本仓库当前的正式目标是：在 **Original DCP** 基础上训练并验证一条已冻结的主线：

```text
Original DCP frozen representation
        ↓
relative importance predictor
        ↓
availability-aware bounded Prompt Adapter
        ↓
direct-task continuous UtilityGate
        ↓
frozen DCP classifier / MM-IMDb BCE
```

工程侧已经完成路径迁移、依赖入口和预检脚本；研究侧仍需在 AIStation 上完成正式训练、消融和多 seed 验证。**当前不能宣称性能提升已经完成**。


## 6. 创建 Python 虚拟环境

平台镜像必须已经自带可用的 GPU PyTorch。安装脚本不会重新安装或替换 `torch` / `torchvision`，只安装项目其余依赖。

优先把虚拟环境放在容器内部，便于后续“保存镜像”：

```bash
export DCP_VENV_DIR=/opt/dcp-venv
bash setup_aistation_env.sh
```

如果 `/opt` 不可写，改用源码目录内环境：

```bash
export DCP_VENV_DIR=.venv-aistation
bash setup_aistation_env.sh
```

安装完成后验证：

```bash
export PYTHON_BIN='$DCP_VENV_DIR/bin/python'
'$PYTHON_BIN' -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

如果平台不能访问公网，先准备 Linux / Python 3.8 对应的 wheelhouse，再执行：

```bash
export DCP_WHEELHOUSE=<wheelhouse 挂载目录>
bash setup_aistation_env.sh
```

不要使用旧 `requirements.txt` 直接安装；它包含历史重复 pin 和不适合平台镜像的旧 CUDA/PyTorch 版本。平台训练一律使用 `requirements_aistation.txt`。


## 7. 运行平台预检

在源码目录执行：

```bash
bash aistation_preflight.sh
```

成功标志：

```text
AISTATION_PREFLIGHT=OK
```

预检会检查：

- Python 版本与关键依赖
- CUDA 是否可用
- 三个 Arrow 文件的行数与字段
- 三个 missing table
- `ViT-B-16.pt`
- Original DCP checkpoint
- `run.py` 与 launcher

如果只是先验证文件，可临时使用：

```bash
bash aistation_preflight.sh --allow-cpu
```

但正式训练环境必须通过 CUDA 检查。


## 8. 保存镜像

只有虚拟环境位于容器文件系统（例如 `/opt/dcp-venv`）时，才能可靠地随镜像保存。用户目录和挂载数据通常不应打进镜像。

操作：

1. 回到开发环境详情页。
2. 单击“保存镜像”。
3. 镜像名称使用小写，例如 `dcp-reliability`。
4. tag 填写 `v1`。
5. 描述填写 `DCP relative importance bounded adapter direct task gate`。
6. 提交保存。
7. 在“镜像管理 > 传输列表”中等待状态为“成功”。
8. 确认个人镜像列表中可见。

保存镜像后不要删除工程目录；训练代码、数据和输出仍依赖平台文件系统挂载。


## 9. 正式训练前的 2-batch 检查

先做跨阶段检查：

```bash
bash run_aistation.sh \
  max_epoch=2 \
  adapter_train_epochs=1 \
  limit_train_batches=2 \
  limit_val_batches=2
```

必须依次看到：

```text
TRAINING_PHASE=ADAPTER
TRAINING_PHASE=GATE
GATE_SUPERVISION=DIRECT_TASK
```

如果出现 CUDA OOM，先降低 micro-batch：

```bash
bash run_aistation.sh \
  per_gpu_batchsize=2 \
  batch_size=64 \
  max_epoch=2 \
  adapter_train_epochs=1 \
  limit_train_batches=2 \
  limit_val_batches=2
```

仍然 OOM 时改为 `per_gpu_batchsize=1`，保持 `batch_size=64`，由梯度累积维持有效 batch。

### 9.1 本机冒烟测试

在 Windows PowerShell 中运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_local_smoke.ps1
```

该脚本会：

- 使用本机 `.conda-dcp-train` 环境
- 执行 2-batch、2-epoch 的跨阶段检查
- 输出到 `experiments\dcp_local_smoke`
- 确认依次出现 `TRAINING_PHASE=ADAPTER`、`TRAINING_PHASE=GATE`、`GATE_SUPERVISION=DIRECT_TASK`


## 10. 创建正式训练任务

进入：

```text
AI 开发 > 模型开发 > 模型训练 > 创建
```

推荐配置：

| 字段 | 建议值 |
| --- | --- |
| 名称 | `dcp_relative_direct_seed0` |
| 镜像 | `dcp-reliability:v1` |
| 部署类型 | 单机 |
| 资源组 | 有空闲 GPU 的资源组 |
| 加速卡 | 1 张，与开发环境相同类型优先 |
| CPU | 8 核起步 |
| 内存 | 32GB 起步 |
| Shm | 8GB 或 16GB |
| 运行方式 | 命令模式 |
| 命令 | `bash run_aistation.sh` |
| 执行目录 | `code/Deep_Correlated_Prompting-main` |
| 数据 | 直接使用工程根目录，或分别挂载所需目录 |
| 日志路径 | `outputs/direct_task_full_logs` |
| 模型输出 | 可开启，路径选择 `outputs/direct_task_full_logs` |

如果虚拟环境保存在镜像内的 `/opt/dcp-venv`，添加环境变量：

```text
DCP_VENV_DIR=/opt/dcp-venv
```

如果目录结构与默认一致，其他路径变量可以不填。如果平台使用独立挂载，则填写：

```text
MMIMDB_DATA_ROOT=<实际 Arrow 挂载目录>
MISSING_TABLE_ROOT=<实际 missing table 挂载目录>
CLIP_CACHE_ROOT=<包含 ViT-B-16.pt 的目录>
ORIGINAL_DCP_PATH=<实际 Original DCP checkpoint 文件>
TRAIN_LOG_DIR=<实际持久化输出目录>
EXP_NAME=relative_bounded_direct_task_20ep_seed0
```

环境变量默认值：

| 变量 | 指向内容 | 默认值 |
| --- | --- | --- |
| `MMIMDB_DATA_ROOT` | 三个 Arrow 文件所在目录 | `../../data/mmimdb/arrow` |
| `MISSING_TABLE_ROOT` | missing table 所在目录 | `../../data/mmimdb/missing_tables` |
| `CLIP_CACHE_ROOT` | `ViT-B-16.pt` 所在目录 | `../../cache/clip` |
| `ORIGINAL_DCP_PATH` | Original DCP `.ckpt` | `../../experiments/dcp_mmimdb_reproduction/checkpoints/dcp_mmimdb_reproduction_seed0_seed0/version_2/checkpoints/epoch=3-step=507.ckpt` |
| `TRAIN_LOG_DIR` | 输出目录 | `../../outputs/direct_task_full_logs` |
| `EXP_NAME` | 实验名 | `relative_bounded_direct_task_20ep` |
| `PYTHON_BIN` | Python 解释器 | `python` |

提交后任务会经历：

```text
排队中 → 运行中 → 完成 / 失败
```


## 11. 正式训练命令

平台任务中的正式命令只有：

```bash
bash run_aistation.sh
```

launcher 已锁定：

- MM-IMDb
- relative importance
- availability-aware bounded Adapter
- direct-task continuous Gate
- Adapter 10 epoch
- Gate 10 epoch
- 总计 20 epoch
- seed 0
- 单 GPU
- FP16

首次正式训练前不要继续修改模型结构或新增损失。


## 12. 训练监控

在“模型开发 > 模型训练”中单击任务名称，查看：

- 基本信息
- 任务日志
- 容器实例
- GPU/CPU/内存指标
- 可视化（如已配置日志路径）

日志重点检查：

```text
epoch 0-9:  TRAINING_PHASE=ADAPTER
epoch 10-19: TRAINING_PHASE=GATE
GATE 阶段:   GATE_SUPERVISION=DIRECT_TASK
```

不应出现：

- `NaN`
- `CUDA out of memory`
- checkpoint path error
- 数据行数或字段不匹配

训练产物位于 `TRAIN_LOG_DIR` 下。可用：

```bash
find $TRAIN_LOG_DIR -type f \( -name 'last.ckpt' -o -name '*.ckpt' \) -print
```


## 13. 训练后评估

进入同一代码执行目录，先恢复与训练一致的环境变量；如果使用独立挂载，请替换为实际路径：

```bash
export MMIMDB_DATA_ROOT=${MMIMDB_DATA_ROOT:-../../data/mmimdb/arrow}
export MISSING_TABLE_ROOT=${MISSING_TABLE_ROOT:-../../data/mmimdb/missing_tables}
export ORIGINAL_DCP_PATH=${ORIGINAL_DCP_PATH:-../../experiments/dcp_mmimdb_reproduction/checkpoints/dcp_mmimdb_reproduction_seed0_seed0/version_2/checkpoints/epoch=3-step=507.ckpt}
export TRAIN_LOG_DIR=${TRAIN_LOG_DIR:-../../outputs/direct_task_full_logs}

if [ -n ${DCP_VENV_DIR:-} ] && [ -x $DCP_VENV_DIR/bin/python ]; then
  export PYTHON_BIN=${PYTHON_BIN:-$DCP_VENV_DIR/bin/python}
else
  export PYTHON_BIN=${PYTHON_BIN:-python}
fi

export FINAL_CKPT=<训练产生的 last.ckpt>
export EVAL_OUTPUT=$TRAIN_LOG_DIR/evaluation_seed0.json
mkdir -p $TRAIN_LOG_DIR
```

运行 validation 评估：

```bash
$PYTHON_BIN tools/evaluate_final_reliability.py \
  $FINAL_CKPT \
  $ORIGINAL_DCP_PATH \
  $EVAL_OUTPUT \
  --split val \
  --data-root $MMIMDB_DATA_ROOT \
  --missing-table-root $MISSING_TABLE_ROOT \
  --seed 0
```

### 13.1 研究成功标准

只有同时满足以下条件，才能声明创新点真正有效：

1. learned continuous 同时优于 `g1` 与 `best-fixed`。
2. learned continuous 优于 shuffled continuous。
3. Gate 输出不是接近常数，样本间有有效差异。
4. predicted importance 优于 condition-shuffled，最好也优于 condition-prior。
5. seed 0 成功后再运行至少 seed 1、2，报告均值和方差。

如果 Adapter 提升但 learned 不优于 shuffled，只能声明 Adapter 创新有效，不能声明重要性驱动 Gate 已有效。

### 13.2 多 seed

seed 0 验证流程正常后，在 AIStation 复制训练任务，只修改任务名与命令：

```bash
EXP_NAME=relative_bounded_direct_task_20ep_seed1 bash run_aistation.sh seed=1
EXP_NAME=relative_bounded_direct_task_20ep_seed2 bash run_aistation.sh seed=2
```

分别把对应 `last.ckpt` 代入评估命令，并把 `EVAL_OUTPUT`、`--seed` 改为 1 或 2。

所有结构与超参数决定都应在 validation 上完成；方案冻结后，再把 `--split val` 改为 `--split test` 做最终一次 test 评估，不要根据 test 结果继续调参。


## 14. 常见错误处理

### `original_dcp_path does not exist`

训练任务没有挂载 checkpoint，或 `ORIGINAL_DCP_PATH` 填错。回开发环境用 `ls -lh` 确认实际路径，再更新训练任务环境变量。

### CLIP 权重尝试联网下载

`CLIP_CACHE_ROOT` 没有指向含 `ViT-B-16.pt` 的目录，或文件名不正确。

### `CUDA is unavailable`

选成了 CPU 镜像、没有申请 GPU，或保存镜像时破坏了平台 CUDA/PyTorch 环境。重新选择平台 GPU PyTorch 镜像。

### 缺少 Python 包

确认训练任务使用保存后的镜像，并设置了 `DCP_VENV_DIR`。在开发环境重新运行：

```bash
bash setup_aistation_env.sh
bash aistation_preflight.sh
```

### CUDA OOM

训练命令末尾追加：

```text
per_gpu_batchsize=2 batch_size=64
```

仍然 OOM 时使用 `per_gpu_batchsize=1`。

### 数据行数或字段不匹配

上传或解压了错误的 Arrow 文件。预检要求：

- train：15552
- dev：2608
- test：7799

字段必须为 `image`、`plots`、`label`、`genres`、`image_id`、`split`。

### 任务名不合法

AIStation 任务名只接受英文字母、数字和下划线，且不能以下划线开头。


## 15. 历史结论与禁止事项

以下结论来自旧审查、实验台账和交接报告：

1. `relative importance` 已有比 absolute 路线更好的相关性和 Adapter headroom。
2. `bounded Adapter` 的有效性证据已被保留。
3. Binary Gate 和五档 hard g-star 不是当前最优默认。
4. `direct_task continuous Gate` 是当前审查后的正式主线。
5. Gate 阶段只训练 UtilityGate；Adapter、Predictor、DCP 主干和 classifier 必须冻结。
6. 旧 Reliability checkpoint 的失败不能代表当前 `direct_task` 主线。
7. 没有完整消融前，不能宣称性能提升或创新点有效。
8. 不要在首次正式训练前继续修改模型结构、新增损失或扩大网络。
9. 平台路径只应通过环境变量注入，不应写死到源码。
10. 历史 checkpoint、日志和 JSON 中的绝对路径属于 provenance，不应批量改写。


## 16. 最短操作清单

如果只看最短流程：

1. 确认文件已上传并解压。
2. 创建 Python 3.8 / GPU PyTorch 单机开发环境。
3. 进入 `code/Deep_Correlated_Prompting-main`。
4. 执行：

```bash
export DCP_VENV_DIR=/opt/dcp-venv
bash setup_aistation_env.sh
bash aistation_preflight.sh
```

5. 保存镜像 `dcp-reliability:v1`。
6. 运行 2-batch 跨阶段检查。
7. 创建训练任务，执行目录选择代码目录，命令填写：

```bash
bash run_aistation.sh
```

8. 训练结束后运行 `tools/evaluate_final_reliability.py`，完成 learned、shuffled、fixed、oracle 和 importance 消融判决。


## 17. 文档合并说明

本文件已合并以下旧文档的核心内容：

- `AISTATION_END_TO_END_GUIDE.md`
- `AISTATION_PORTABLE_RUN.md`
- `DCP_AI_HANDOFF_2026-08-29.md`
- `DCP_INNOVATION_CODE_AUDIT_2026-08-29.md`
- `DCP_INNOVATION_EXPERIMENT_LEDGER_2026-08-29.md`
- `DCP_INNOVATION_REFACTOR_PLAN_2026-08-29.md`
- `dcp_reliability_comparison_report.md`
- `DCP_RELIABILITY_PROVENANCE_AUDIT_2026-08-15.md`

同时结合了《AIStation 人工智能平台 V5.0 普通用户手册》中关于文件管理、数据集管理、镜像管理、开发环境、模型训练和监控的章节。
