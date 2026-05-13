# 修复方案: DeepSpeed `auto` 值无法解析

## 错误日志

```
ValueError: Can't find `model.config` entry, therefore it's not possible to automatically
fill out the following `auto` entries in the DeepSpeed config file:
['zero_optimization.reduce_bucket_size'].
```

## 根本原因

自定义 `GPT` 模型 (`modules/gpt.py`) 没有 `model.config` 属性（它不是 HuggingFace `PreTrainedModel`），而 `ds_zero2.json` 中存在大量 `"auto"` 值。

当 `accelerator.prepare()` 被调用时，accelerate 尝试通过 `model.config` 自动推导这些 `"auto"` 值 → 找不到 → 抛出 `ValueError`。

`ds_zero2.json` 中所有 `"auto"` 字段：

| 字段 | 需要改为 |
|---|---|
| `optimizer.params.lr` | 已知静态值 |
| `optimizer.params.weight_decay` | 已知静态值 |
| `scheduler.params.warmup_min_lr` | 已知静态值 |
| `scheduler.params.warmup_max_lr` | 已知静态值 |
| `scheduler.params.warmup_num_steps` | **运行时动态计算** |
| `scheduler.params.total_num_steps` | **运行时动态计算** |
| `zero_optimization.reduce_bucket_size` | 已知常量 |
| `gradient_accumulation_steps` | 已知静态值 |
| `gradient_clipping` | 已知静态值 |
| `train_batch_size` | 可移除（有显式 gradient_accumulation_steps 时不需要） |
| `train_micro_batch_size_per_gpu` | 可移除 |

---

## 修复一：`ds_zero2.json` — 将所有 `"auto"` 替换为具体值

```json
{
    "fp16": {
        "enabled": true
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2.5e-4,
            "weight_decay": 0.01
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0.0,
            "warmup_max_lr": 2.5e-4,
            "warmup_num_steps": 0,
            "total_num_steps": 0
        }
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 200000000,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 500000000,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0
}
```

**说明：**
- `lr`、`weight_decay`、`warmup_min_lr`、`warmup_max_lr`：来自 `config.py` 中的静态值
- `reduce_bucket_size`：设为 `5e8`（DeepSpeed ZeRO-2 的常用默认值）
- `gradient_accumulation_steps`：来自 `config.PretrainConfig.accumulate_grad_batches`（值为 4）
- `gradient_clipping`：来自 `config.clip`（值为 1）
- `warmup_num_steps` / `total_num_steps`：先填占位值 `0`，运行时由代码动态注入（见修复二）
- 移除了 `train_batch_size` 和 `train_micro_batch_size_per_gpu`（有显式 `gradient_accumulation_steps` 时不需要）

---

## 修复二：`accelerate_pretrain.py` — 动态注入 `warmup_num_steps` / `total_num_steps`

这两个值是运行时动态计算的（依赖 `loading_ratio`、dataloader 长度等），无法写死在 JSON 里。需要在代码中计算后注入到 DeepSpeed 配置。

**当前代码结构（第 83-132 行，仅保留轮廓）：**

```python
# Line 83-86: 获取 ds_config
if accelerator.state.deepspeed_plugin is not None:
    ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
else:
    ds_config = None

# Line 91-102: 创建 optimizer

# Line 104-114: 计算 total_steps 和 warmup_steps

# Line 117-128: 创建 scheduler

# Line 130-132: accelerator.prepare()
```

**问题：** `warmup_num_steps` 和 `total_num_steps` 在 `prepare()` 时就要被 DeepSpeed 使用，但计算发生在 optimizer 之后、scheduler 之前。只需在创建 scheduler 之前，把计算好的值注入 `ds_config`。

**修改：在第 114 行后（计算完 warmup_steps 后）插入一段注入逻辑：**

```python
# 在 Line 114 之后插入
# 将运行时计算的值注入 DeepSpeed 配置，替代 JSON 中的占位值
if ds_config is not None and "scheduler" in ds_config:
    ds_config["scheduler"]["params"]["warmup_num_steps"] = warmup_steps
    ds_config["scheduler"]["params"]["total_num_steps"] = total_steps
```

---

## 改动总览

| 文件 | 改动 |
|---|---|
| `config_files/ds_zero2.json` | 所有 `"auto"` → 具体值（静态值直接用 config.py 中的定义，动态值用 `0` 占位） |
| `accelerate_pretrain.py` | 在 `prepare()` 之前，将运行时计算的 `warmup_steps` / `total_steps` 注入 `ds_config` |
| `config_files/zero2.yaml` | 删除 `mixed_precision: fp16`（上一轮修复，由 ds_zero2.json 中的 `fp16.enabled` 接管） |
