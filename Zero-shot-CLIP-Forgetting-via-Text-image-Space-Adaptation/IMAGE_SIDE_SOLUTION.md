# 图像侧遗忘方案 - 基于Qwen3 ViT + Merger架构

## 你的实际架构

```
图像 → Qwen3 ViT → 最后一层特征 → Merger (LoRA) → 图像嵌入
```

## 与CLIP文本侧的类比

### CLIP文本侧（原始方案）
```
文本 → CLIP Text Encoder → ln_final → text_projection (LoRA) → 文本嵌入
      [Transformer]         [512维]    [512→512]              [512维]
```

### 你的图像侧（目标方案）
```
图像 → Qwen3 ViT → 最后一层 → Merger (LoRA) → 图像嵌入
      [视觉编码器]  [D维特征]  [D→E维]        [E维]
```

---

## 核心映射关系

| 组件 | CLIP文本侧 | 你的图像侧 |
|------|-----------|-----------|
| **编码器** | CLIP Text Encoder | Qwen3 ViT |
| **Hook提取层** | `ln_final` | Qwen3 ViT的最后一层 |
| **Hook特征维度** | 512维 | D维（取决于Qwen3 ViT配置） |
| **修改的层** | `text_projection` | `Merger` |
| **LoRA应用位置** | text_projection | Merger |
| **输出维度** | 512维 | E维（取决于Merger输出） |

---

## 实现方案

### 1. Hook机制 - 提取Qwen3 ViT最后一层特征

```python
hooks = {}

def get_qwen_vit_activation(name):
    """提取Qwen3 ViT最后一层的输出"""
    def hook(model, input, output):
        global hooks
        # 假设最后一层名称为 'final_layer' 或类似
        # 需要根据实际模型结构确定
        if name == 'final_layer':  # 替换为实际层名
            hooks[name] = output.detach()
    return hook

def register_qwen_vit_hooks(qwen_vit_model):
    """给Qwen3 ViT注册hook"""
    for name, module in qwen_vit_model.named_modules():
        module._forward_hooks = OrderedDict()

    for name, module in qwen_vit_model.named_modules():
        module.register_forward_hook(get_qwen_vit_activation(name))
```

**关键问题需要确认**:
- Qwen3 ViT的最后一层叫什么名字？
- 最后一层输出的维度是多少？（D维）

---

### 2. 预计算图像特征

```python
@torch.no_grad()
def precompute_image_features(qwen_vit, merger, forget_dataloader, device):
    """
    类比CLIP文本侧的 precompute_projections

    输入: 图像数据加载器
    输出:
    - final_embeddings: Merger的输出 [N, E]
    - vit_features: Qwen3 ViT最后一层的特征 [N, D]
    """
    global hooks
    vit_features = []
    final_embeddings = []

    for batch in forget_dataloader:
        hooks = {}
        images = batch['images'].to(device)

        # 前向传播
        vit_output = qwen_vit(images)  # 触发hook
        embedding = merger(vit_output)  # 通过Merger

        final_embeddings.append(embedding)
        vit_features.append(hooks['final_layer'].clone())  # 从hook提取

    return torch.cat(final_embeddings), torch.cat(vit_features)
```

**类比**:
- CLIP文本侧: `hooks['ln_final']` → 你的: `hooks['final_layer']`
- CLIP文本侧: `model.encode_text()` → 你的: `qwen_vit() + merger()`

---

### 3. 计算中性目标

```python
@torch.no_grad()
def compute_neutral_target(num_forget, output_dim, device):
    """
    类比CLIP文本侧的 compute_proj_into

    CLIP文本侧: 使用空字符串 "" 的嵌入
    你的图像侧: 使用随机向量
    """
    # 方法1: 纯随机向量（推荐）
    proj_into = torch.randn(num_forget, output_dim).to(device)
    proj_into = proj_into / proj_into.norm(dim=-1, keepdim=True)

    return proj_into
```

**说明**:
- `output_dim` = Merger的输出维度（E维）
- 归一化到单位球面，与CLIP保持一致

---

### 4. 给Merger添加LoRA

```python
from utils_lora import Linear

def create_lora_merger(original_merger, lora_r=5, device='cuda'):
    """
    类比CLIP文本侧创建 new_text_proj

    将原始Merger替换为带LoRA的版本
    """
    # 获取Merger的输入输出维度
    # 假设Merger是一个Linear层: [D, E]
    in_dim = original_merger.weight.shape[1]   # D维
    out_dim = original_merger.weight.shape[0]  # E维

    # 创建LoRA层
    lora_merger = Linear(
        in_features=in_dim,
        out_features=out_dim,
        r=lora_r,
        bias=False,
        device=device
    )

    # 初始化为原始权重（冻结）
    lora_merger.weight = nn.Parameter(original_merger.weight.T)
    lora_merger.weight.requires_grad = False

    return lora_merger
```

**关键问题需要确认**:
- Merger的结构是什么？单层Linear还是多层MLP？
- 如果是多层，在哪一层添加LoRA？

---

### 5. 损失函数（与CLIP完全相同）

```python
def train_image_forgetting(
    qwen_vit,
    original_merger,
    forget_dataloader,
    preserve_dataloader,
    device,
    lora_r=5,
    lamb_forget=1.3,
    lamb_preserve=0.4,
    lamb_weight=1.0,
    epochs=2000,
    lr=0.01
):
    # 1. 注册hook
    register_qwen_vit_hooks(qwen_vit)

    # 2. 预计算特征
    forget_embeddings, forget_vit_features = precompute_image_features(
        qwen_vit, original_merger, forget_dataloader, device
    )
    preserve_embeddings, preserve_vit_features = precompute_image_features(
        qwen_vit, original_merger, preserve_dataloader, device
    )

    # 3. 计算中性目标
    output_dim = forget_embeddings.shape[1]  # E维
    proj_into = compute_neutral_target(
        num_forget=forget_vit_features.shape[0],
        output_dim=output_dim,
        device=device
    )

    # 4. 创建LoRA Merger
    lora_merger = create_lora_merger(original_merger, lora_r, device)
    optimizer = torch.optim.Adam(lora_merger.parameters(), lr=lr)

    # 5. 训练循环
    best_loss = np.inf
    best_weights = None

    for epoch in range(epochs):
        # 前向传播（使用预提取的ViT特征）
        new_forget_output = lora_merger(forget_vit_features)
        new_preserve_output = lora_merger(preserve_vit_features)

        # 计算损失（与CLIP文本侧完全相同）
        forget_loss = lamb_forget * torch.norm(
            proj_into - new_forget_output, p=2
        )
        preserve_loss = lamb_preserve * torch.norm(
            preserve_embeddings - new_preserve_output, p=2
        )
        weight_loss = lamb_weight * torch.norm(
            (lora_merger.lora_B @ lora_merger.lora_A).T, p=2
        )

        loss = forget_loss + preserve_loss + weight_loss

        # 保存最佳权重
        if loss < best_loss:
            best_loss = loss
            best_weights = lora_merger.weight.clone()

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: forget={forget_loss.item():.4f}, "
                  f"preserve={preserve_loss.item():.4f}, "
                  f"total={loss.item():.4f}")

    # 6. 更新Merger
    original_merger.weight = nn.Parameter(best_weights.T)

    return original_merger
```

---

## 完整数据流

### CLIP文本侧
```
"dog"
  ↓ tokenize
[token_ids]
  ↓ Text Encoder
ln_final特征 [512] ← Hook提取
  ↓ text_projection (LoRA)
文本嵌入 [512]
```

### 你的图像侧
```
dog.jpg
  ↓ 预处理
[3, 224, 224]
  ↓ Qwen3 ViT
最后一层特征 [D] ← Hook提取
  ↓ Merger (LoRA)
图像嵌入 [E]
```

---

## 损失函数对比

### CLIP文本侧
```python
# 输入
change_hooks = ln_final特征 [N_forget, 512]
preserve_hooks = ln_final特征 [N_preserve, 512]
proj_into = 空字符串嵌入 [N_forget, 512]

# 损失
forget_loss = ||proj_into - new_text_proj(change_hooks)||₂
preserve_loss = ||preserve_output - new_text_proj(preserve_hooks)||₂
weight_loss = ||lora_B @ lora_A||₂
```

### 你的图像侧
```python
# 输入
forget_vit_features = Qwen3 ViT特征 [N_forget, D]
preserve_vit_features = Qwen3 ViT特征 [N_preserve, D]
proj_into = 随机向量 [N_forget, E]

# 损失（公式完全相同）
forget_loss = ||proj_into - lora_merger(forget_vit_features)||₂
preserve_loss = ||preserve_embeddings - lora_merger(preserve_vit_features)||₂
weight_loss = ||lora_B @ lora_A||₂
```

---

## 需要你确认的信息

### 1. Qwen3 ViT结构
```python
# 最后一层的名称是什么？
# 例如: 'norm', 'ln_post', 'final_norm', 'layer_norm'?

# 如何获取？
for name, module in qwen_vit.named_modules():
    print(name)  # 找到最后一层的名称
```

### 2. Merger结构
```python
# Merger是什么结构？
# 选项A: 单层Linear
merger = nn.Linear(D, E)

# 选项B: 多层MLP
merger = nn.Sequential(
    nn.Linear(D, hidden),
    nn.ReLU(),
    nn.Linear(hidden, E)
)

# 如果是多层，在哪一层加LoRA？
# 推荐: 最后一层
```

### 3. 维度信息
- Qwen3 ViT最后一层输出维度 D = ?
- Merger输出维度 E = ?

---

## 实现步骤

1. **确认架构信息**（上面3个问题）
2. **修改hook提取层名称**（替换 `'final_layer'`）
3. **确认Merger结构并添加LoRA**
4. **准备数据集**（遗忘类别 + 保留类别）
5. **运行训练**
6. **验证效果**

---

## 与CLIP文本侧的核心区别

| 方面 | CLIP文本侧 | 你的图像侧 |
|------|-----------|-----------|
| **编码器** | CLIP Text Encoder | Qwen3 ViT |
| **输入** | 文本（零样本） | 图像（需要数据） |
| **Hook层** | ln_final | Qwen3 ViT最后一层 |
| **修改层** | text_projection | Merger |
| **中性目标** | 空字符串嵌入 | 随机向量 |
| **损失函数** | ✅ 完全相同 | ✅ 完全相同 |
| **LoRA应用** | ✅ 完全相同 | ✅ 完全相同 |

---

## 总结

**核心思想**: 完全复用CLIP文本侧的损失函数和LoRA训练流程，只需要：
1. 把hook从 `ln_final` 改到 Qwen3 ViT最后一层
2. 把LoRA从 `text_projection` 加到 `Merger`
3. 把输入从文本改为图像

**损失函数不变**: 三组件损失（遗忘 + 保留 + 正则化）完全相同！
