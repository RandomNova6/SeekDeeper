# 修复方案: DeepSpeed 配置冲突

## 错误原因

当 `zero2.yaml` 中指定了 `deepspeed_config_file` 时，accelerate 要求以下配置项必须在 **DeepSpeed 自身的配置文件**（`ds_zero2.json`）中设置，而不能出现在 accelerate 的配置（YAML）或 `Accelerator()` 构造函数参数中：

- `gradient_accumulation_steps`
- `gradient_clipping`
- `zero_stage` / offload 相关
- `mixed_precision`

当前代码在两个地方违反了这一规则：

| 位置 | 问题 |
|---|---|
| `accelerate_pretrain.py:58` | `Accelerator()` 构造函数传入了 `gradient_accumulation_steps` |
| `zero2.yaml:11` | 设置了 `mixed_precision: fp16`（已被 `ds_zero2.json` 中的 `bf16` 覆盖） |

---

## 修复方案

### 1. 修改 `accelerate_pretrain.py`

在 `Accelerator()` 构造函数中去掉 `gradient_accumulation_steps` 参数，改为在构造完成后根据是否使用 DeepSpeed 分别处理：

```python
# 修改前 (第 56-60 行)
accelerator = Accelerator(
    project_config=config_project,
    gradient_accumulation_steps=config.PretrainConfig.accumulate_grad_batches,
    kwargs_handlers=[kwargs]
)

# 修改后
accelerator = Accelerator(
    project_config=config_project,
    kwargs_handlers=[kwargs]
)

# 非 DeepSpeed 模式下，手动设置 gradient_accumulation_steps
# DeepSpeed 模式下，该值由 ds_zero2.json 中的配置自动决定
if accelerator.state.deepspeed_plugin is None:
    accelerator.gradient_accumulation_steps = config.PretrainConfig.accumulate_grad_batches
```

**为什么这样改：**
- DeepSpeed 模式下，`gradient_accumulation_steps` 由 `ds_zero2.json` 中的 `"gradient_accumulation_steps": "auto"` 控制，accelerate 会自动解析
- 非 DeepSpeed 模式（single_gpu / multi_gpu）下，手动设置该属性以保证 `total_steps`、`warmup_steps` 等后续计算正确

---

### 2. 修改 `config_files/zero2.yaml`

去掉 `mixed_precision` 行，因为混合精度已在 `ds_zero2.json` 中通过 `"bf16": {"enabled": true}` 配置：

```yaml
# 修改前
mixed_precision: fp16

# 修改后
# 删除此行（由 ds_zero2.json 中的 bf16 配置接管）
```

修改后的 `zero2.yaml` 第 11 行附近：

```yaml
main_training_function: main
# mixed_precision 已移入 ds_zero2.json
num_machines: 1
```

---

## 改动总览

| 文件 | 改动内容 |
|---|---|
| `accelerate_pretrain.py` | 从 `Accelerator()` 构造中移除 `gradient_accumulation_steps`，改为在非 DeepSpeed 下手动赋值 |
| `config_files/zero2.yaml` | 删除 `mixed_precision: fp16` |
| `config_files/ds_zero2.json` | 无需修改（已正确配置 `gradient_accumulation_steps`、`gradient_clipping`、`bf16`） |
| `config_files/single_gpu.yaml` | 无需修改（不使用 deepspeed，且其 `mixed_precision: fp16` 不受影响） |
| `config_files/multi_gpu.yaml` | 无需修改（同上） |
