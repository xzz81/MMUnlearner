# unlearner_v2 测试指南

## 测试目标

验证 **finetune → manifold unlearn → eval** 三阶段核心流程的正确性。

## 前置检查

测试前确认以下路径存在：

```bash
# 检查模型
ls /home/dcy/project/MMUnlearner/models/Qwen/Qwen2-VL-7B-Instruct/

# 检查数据集
ls /home/dcy/project/MMUnlearner/data/MLLMU-Bench/

# 检查配置文件
ls unlearner_v2/config/*.yaml
```

## 配置调整

### 1. 修改 `unlearner_v2/config/data.yaml`

```yaml
batch_size: 4  # 从 2 改为 4（匹配原始参数）
```

### 2. 修改 `unlearner_v2/config/train.yaml`

```yaml
num_epochs: 5  # 从 1 改为 5（匹配原始参数）
```

### 3. 修改 `unlearner_v2/config/config.yaml`（按阶段调整）

**阶段 1 - Finetune**:
```yaml
finetune_phase:
  enable: true
unlearn_phase:
  enable: false
eval_phase:
  enable: false
```

**阶段 2 - Unlearn**:
```yaml
finetune_phase:
  enable: false
unlearn_phase:
  enable: true
  method: "manifold"
eval_phase:
  enable: false
```

**阶段 3 - Eval**:
```yaml
finetune_phase:
  enable: false
unlearn_phase:
  enable: false
eval_phase:
  enable: true
```

## 测试流程

### 阶段 1: Finetune 测试

**目标**: 验证模型加载、LoRA 应用、训练流程

**步骤**:
```bash
# 1. 配置 config.yaml（仅启用 finetune）
vim unlearner_v2/config/config.yaml

# 2. 运行训练
python unlearner_v2/main.py --config unlearner_v2/config

# 3. 检查输出
ls unlearner_v2/checkpoints/finetune/
```

**成功标准**:
- ✅ 模型成功加载
- ✅ LoRA 模块正确应用
- ✅ 训练完成 5 个 epoch
- ✅ 训练 loss 持续下降
- ✅ checkpoint 保存成功
- ✅ 无 CUDA OOM 错误（如有，降低 batch_size 到 2）

**预期输出**:
- Checkpoint: `unlearner_v2/checkpoints/finetune/epoch=4-step=XXX.ckpt`

---

### 阶段 2: Unlearn 测试

**目标**: 验证 Manifold 方法的梯度上升遗忘

**前置条件**: 阶段 1 的 finetune checkpoint 存在

**步骤**:
```bash
# 1. 配置 config.yaml（仅启用 unlearn）
vim unlearner_v2/config/config.yaml

# 2. 确认 checkpoint 路径（如需要，更新 model.yaml 或 train.yaml）
# 系统应自动加载 finetune checkpoint

# 3. 运行 unlearn
python unlearner_v2/main.py --config unlearner_v2/config

# 4. 检查输出
ls unlearner_v2/checkpoints/unlearn/
```

**成功标准**:
- ✅ 正确加载 finetune checkpoint
- ✅ 交替执行 forget/retain 批次
- ✅ Forget loss 上升（梯度上升生效）
- ✅ Retain loss 保持稳定
- ✅ Checkpoint 保存成功

**预期输出**:
- Checkpoint: `unlearner_v2/checkpoints/unlearn/epoch=4-step=XXX.ckpt`

**可选**: 测试 grad_mask
```bash
# 生成 mask（如需要）
python data_process/MLLMU_gen_mask.py

# 在 train.yaml 中设置
# grad_mask_path: "/path/to/mask.pt"
```

---

### 阶段 3: Eval 测试

**目标**: 验证评估指标计算和遗忘效果

**前置条件**: 阶段 1 和 2 的 checkpoint 都存在

**步骤**:
```bash
# 1. 配置 config.yaml（仅启用 eval）
vim unlearner_v2/config/config.yaml

# 2. 运行评估
python unlearner_v2/main.py --config unlearner_v2/config

# 3. 检查结果
cat unlearner_v2/output/eval_results.json
```

**成功标准**:
- ✅ 正确加载两个 checkpoint（finetune + unlearn）
- ✅ 生成所有测试样本的预测
- ✅ 计算 BLEU 和 ROUGE 指标
- ✅ 结果保存为 JSON
- ✅ 观察到遗忘行为：
  - **Forget set**: BLEU/ROUGE 分数下降
  - **Retain set**: BLEU/ROUGE 分数保持

**预期输出**:
- 结果文件: `unlearner_v2/output/eval_results.json`

---

## 原始参数对照表

| 参数 | 原始值 | 当前值 | 状态 |
|------|--------|--------|------|
| batch_size | 4 | 2 → 4 | ⚠️ 需修改 |
| num_epochs | 5 | 1 → 5 | ⚠️ 需修改 |
| lr | 5e-4 | 5e-4 | ✅ 匹配 |
| LoRA r | 16 | 16 | ✅ 匹配 |
| LoRA alpha | 16 | 16 | ✅ 匹配 |
| forget_alpha | 1.0 | 1.0 | ✅ 匹配 |
| forget_split_ratio | 5% | 5% | ✅ 匹配 |
| gradient_clip | 1.0 | 1.0 | ✅ 匹配 |

## 常见问题排查

### 问题 1: CUDA OOM
**现象**: 训练时显存不足
**解决**:
```yaml
# data.yaml
batch_size: 2  # 降低到 2
```

### 问题 2: Checkpoint 路径错误
**现象**: Unlearn/Eval 阶段找不到 checkpoint
**解决**: 检查 `unlearner_v2/checkpoints/` 目录，确认上一阶段成功完成

### 问题 3: 数据集加载失败
**现象**: 找不到 MLLMU 数据集
**解决**:
```yaml
# data.yaml
data_dir: "/home/dcy/project/MMUnlearner/data/MLLMU-Bench"  # 确认路径正确
```

### 问题 4: LoRA 未应用
**现象**: 训练速度慢或显存占用高
**解决**:
```yaml
# model.yaml
lora:
  enable: true  # 确认已启用
```

### 问题 5: 未观察到遗忘效果
**现象**: Forget set 分数未下降
**排查**:
- 检查 `forget_alpha: 1.0` 是否为正值
- 查看训练日志确认交替更新正常
- 确认评估时使用了正确的 checkpoint

## grad_mask 功能说明

### 状态: ✅ 已完整实现

**功能**:
- 加载 `.pt` 格式的 mask 文件
- 自动过滤 proj/fc/linear/mlp/qkv 层
- 在优化器更新前应用 mask

**使用方法**:
```bash
# 1. 生成 mask（可选）
python data_process/MLLMU_gen_mask.py

# 2. 配置路径
# train.yaml
unlearn:
  manifold:
    grad_mask_path: "/path/to/grad_mask.pt"

# 3. 运行 unlearn 阶段
python unlearner_v2/main.py --config unlearner_v2/config
```

## 测试完成后

### 如果测试通过
1. 更新 `unlearner_v2/to do.md`，标记 Unlearn 和 Eval 阶段为已完成
2. 对比原始 baseline 的结果，验证性能一致性
3. 可选：测试其他模型（Qwen3-VL, LLaVA）

### 如果发现问题
1. 记录具体错误信息和堆栈跟踪
2. 检查日志文件定位问题阶段
3. 参考"常见问题排查"部分

## 快速测试命令汇总

```bash
# 完整测试流程
cd /home/dcy/project/MMUnlearner

# 阶段 1: Finetune
# 修改 config.yaml: finetune=true, 其他=false
python unlearner_v2/main.py --config unlearner_v2/config

# 阶段 2: Unlearn
# 修改 config.yaml: unlearn=true, 其他=false
python unlearner_v2/main.py --config unlearner_v2/config

# 阶段 3: Eval
# 修改 config.yaml: eval=true, 其他=false
python unlearner_v2/main.py --config unlearner_v2/config

# 查看结果
cat unlearner_v2/output/eval_results.json
```

## 预期时间估算

- **Finetune**: ~数小时（取决于 GPU 和数据量）
- **Unlearn**: ~数小时
- **Eval**: ~30分钟-1小时

总计: 约 **半天到一天**（使用单卡 GPU）
