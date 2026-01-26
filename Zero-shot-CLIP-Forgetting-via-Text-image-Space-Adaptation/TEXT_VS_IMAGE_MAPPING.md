# 文本侧 vs 图像侧实现对照表

## 核心对应关系

| 组件 | 文本侧 | 图像侧 |
|------|--------|--------|
| **输入数据** | 类别名称字符串 `["dog", "cat"]` | 图像张量 `[N, 3, 224, 224]` |
| **编码器** | `model.encode_text()` | `model.encode_image()` |
| **Hook提取层** | `ln_final` (文本Transformer最后的LayerNorm) | `ln_post` (ViT) 或 `attnpool` (ResNet) |
| **Hook特征维度** | `[N, 512]` | `[N, 768]` (ViT) 或 `[N, 2048]` (ResNet) |
| **修改的投影层** | `model.text_projection` | `model.visual.proj` (ViT) 或 `model.visual.attnpool.c_proj` (ResNet) |
| **投影层维度** | `[512, 512]` | `[768, 512]` (ViT) 或 `[2048, 512]` (ResNet) |
| **中性目标** | 空字符串 `""` 的嵌入 | 随机噪声向量或噪声图像的嵌入 |
| **LoRA秩** | `r=5` | `r=5` |
| **遗忘权重** | `lamb_forget=1.1-1.3` | `lamb_forget=1.1-1.3` |
| **保留权重** | `lamb_preserve=0.25-0.4` | `lamb_preserve=0.25-0.4` |

---

## 代码映射详解

### 1. Hook注册

#### 文本侧 (main.py:127-132)
```python
def register_model_hooks(model):
    for name, module in model.named_modules():
        module.register_forward_hook(get_activation_proj(name))

def get_activation_proj(name):
    def hook(model, input, output):
        if name == 'ln_final':  # 文本编码器最后的LayerNorm
            hooks[name] = (input[0].detach(), output)
    return hook
```

#### 图像侧 (main_image_side.py)
```python
def register_visual_hooks(model):
    for name, module in model.visual.named_modules():
        module.register_forward_hook(get_activation_visual(name))

def get_activation_visual(name):
    def hook(model, input, output):
        if name in ['ln_post', 'attnpool']:  # 图像编码器最后的层
            hooks[name] = output.detach()
    return hook
```

**关键差异**:
- 文本侧: 遍历整个模型 `model.named_modules()`
- 图像侧: 只遍历视觉编码器 `model.visual.named_modules()`
- Hook目标层不同: `ln_final` vs `ln_post/attnpool`

---

### 2. 预计算特征

#### 文本侧 (main.py:107-123)
```python
@torch.no_grad()
def precompute_projections(model, classes, template=['{}'])):
    list_hooks = []
    projections = []

    for classname in classes:
        hooks = {}
        # 输入: 类别名称
        texts = clip.tokenize([f"a photo of a {classname}"])

        # 前向传播
        class_embeddings = model.encode_text(texts)
        projections.append(class_embeddings)

        # 提取hook特征
        list_hooks.append(hooks['ln_final'][1].clone())

    return torch.cat(projections), torch.cat(list_hooks)
```

#### 图像侧 (main_image_side.py)
```python
@torch.no_grad()
def precompute_image_features(model, image_dataloader, device):
    list_hooks = []
    projections = []

    for batch in image_dataloader:
        hooks = {}
        # 输入: 图像张量
        images = batch['images'].to(device)

        # 前向传播
        image_embeddings = model.encode_image(images)
        projections.append(image_embeddings)

        # 提取hook特征
        if 'ln_post' in hooks:
            list_hooks.append(hooks['ln_post'].clone())
        elif 'attnpool' in hooks:
            list_hooks.append(hooks['attnpool'].clone())

    return torch.cat(projections), torch.cat(list_hooks)
```

**关键差异**:
- 文本侧: 输入是字符串列表，逐个处理
- 图像侧: 输入是DataLoader，批量处理
- 文本侧: 使用模板 `"a photo of a {classname}"`
- 图像侧: 直接使用原始图像

---

### 3. 计算中性目标

#### 文本侧 (main.py:145-157)
```python
@torch.no_grad()
def compute_proj_into(model, idx_cls_forget, ...):
    # 使用空字符串作为中性目标
    empty_text = clip.tokenize("")
    embed = model.encode_text(empty_text)
    proj_into = embed / embed.norm(dim=-1, keepdim=True)

    # 确保中性目标不会被误分类为任何已知类别
    while (original_projections_norm @ proj_into.T).argmax(0) in idx_cls_forget:
        embed = model.encode_text(clip.tokenize(""))
        embed += torch.randn(embed.size()) * 0.5  # 添加噪声
        proj_into = embed / embed.norm(dim=-1, keepdim=True)

    return proj_into
```

#### 图像侧 (main_image_side.py)
```python
@torch.no_grad()
def compute_neutral_image_target(model, num_forget_classes, device, method='random'):
    if method == 'random':
        # 方法1: 纯随机向量（最简单）
        proj_into = torch.randn(num_forget_classes, 512).to(device)
        proj_into = proj_into / proj_into.norm(dim=-1, keepdim=True)

    elif method == 'noise_image':
        # 方法2: 噪声图像的嵌入（更接近文本侧逻辑）
        noise_image = torch.randn(num_forget_classes, 3, 224, 224).to(device)
        noise_image = (noise_image - noise_image.min()) / (noise_image.max() - noise_image.min())

        embed = model.encode_image(noise_image)
        proj_into = embed / embed.norm(dim=-1, keepdim=True)

    return proj_into
```

**关键差异**:
- 文本侧: 有自然的"空"概念（空字符串）
- 图像侧: 需要人为定义"中性"（随机向量或噪声图像）
- 文本侧: 有验证逻辑确保中性目标不会被误分类
- 图像侧: 可以添加类似的验证逻辑

---

### 4. 创建LoRA层

#### 文本侧 (main.py:278-288)
```python
# 获取text_projection的维度
in_proj, out_proj = model.text_projection.shape  # [512, 512]

# 创建LoRA层
new_text_proj = Linear(in_proj, out_proj, r=r, bias=False, device=device)

# 初始化为原始权重（冻结）
new_text_proj.weight = torch.nn.Parameter(model.text_projection.T)
new_text_proj.weight.requires_grad = False
```

#### 图像侧 (main_image_side.py)
```python
# 获取visual projection的维度
if hasattr(model.visual, 'proj'):  # ViT
    visual_proj = model.visual.proj  # [768, 512]
    in_proj, out_proj = visual_proj.shape
else:  # ResNet
    visual_proj = model.visual.attnpool.c_proj.weight  # [512, 2048]
    out_proj, in_proj = visual_proj.shape

# 创建LoRA层
new_visual_proj = Linear(in_proj, out_proj, r=r, bias=False, device=device)

# 初始化为原始权重（冻结）
new_visual_proj.weight = nn.Parameter(visual_proj.T)
new_visual_proj.weight.requires_grad = False
```

**关键差异**:
- 文本侧: 投影层结构统一
- 图像侧: 需要区分ViT和ResNet架构
- 维度不同: 文本侧 `[512, 512]`，图像侧 `[768, 512]` 或 `[2048, 512]`

---

### 5. 损失计算

#### 文本侧 (main.py:297-304)
```python
for epoch in range(EPOCHS):
    # 前向传播
    new_preserve_output = new_text_proj(preserve_hooks)
    new_forget_output = new_text_proj(change_hooks)

    # 计算损失
    forget_loss = lamb_forget * torch.norm(proj_into - new_forget_output, p=2)
    preserve_loss = lamb_preserve * torch.norm(preserve_output - new_preserve_output, p=2)
    weight_loss = lamb_weight * torch.norm((new_text_proj.lora_B @ new_text_proj.lora_A).T, p=2)

    loss = forget_loss + preserve_loss + weight_loss
```

#### 图像侧 (main_image_side.py)
```python
for epoch in range(epochs):
    # 前向传播
    new_preserve_output = new_visual_proj(preserve_hooks)
    new_forget_output = new_visual_proj(forget_hooks)

    # 计算损失（完全相同的公式）
    forget_loss = lamb_forget * torch.norm(proj_into - new_forget_output, p=2)
    preserve_loss = lamb_preserve * torch.norm(preserve_output - new_preserve_output, p=2)
    weight_loss = lamb_weight * torch.norm((new_visual_proj.lora_B @ new_visual_proj.lora_A).T, p=2)

    loss = forget_loss + preserve_loss + weight_loss
```

**关键差异**:
- **损失公式完全相同！**
- 只是输入数据来源不同（文本特征 vs 图像特征）

---

### 6. 更新模型

#### 文本侧 (main.py:332-334)
```python
# 保存最佳权重
new_weights["proj_weight"] = torch.nn.Parameter(best_weights)

# 更新模型
model.load_state_dict({
    **model.state_dict(),
    'text_projection': new_weights["proj_weight"].T
})
```

#### 图像侧 (main_image_side.py)
```python
# 更新模型
if proj_type == 'vit':
    # ViT: 更新 visual.proj
    model.visual.proj = nn.Parameter(best_weights.T)
else:
    # ResNet: 更新 visual.attnpool.c_proj.weight
    model.visual.attnpool.c_proj.weight = nn.Parameter(best_weights.T)
```

**关键差异**:
- 文本侧: 使用 `load_state_dict` 更新
- 图像侧: 直接替换参数
- 图像侧需要区分架构类型

---

## 数据流对比

### 文本侧
```
类别名称 "dog"
    ↓
clip.tokenize("a photo of a dog")
    ↓
Text Encoder (Transformer × 12)
    ↓
ln_final [1, 512] ← Hook提取
    ↓
text_projection (原始) [512, 512]
    ↓
LoRA适配: + lora_B @ lora_A
    ↓
最终文本嵌入 [1, 512]
```

### 图像侧
```
图像 dog.jpg
    ↓
预处理 [1, 3, 224, 224]
    ↓
Image Encoder (ViT × 12 或 ResNet)
    ↓
ln_post [1, 768] 或 attnpool [1, 2048] ← Hook提取
    ↓
visual.proj (原始) [768, 512] 或 [2048, 512]
    ↓
LoRA适配: + lora_B @ lora_A
    ↓
最终图像嵌入 [1, 512]
```

---

## 实现建议

### 1. 数据准备
```python
# 需要准备两个数据集
forget_dataset = YourDataset(
    images=[dog1.jpg, dog2.jpg, ...],
    labels=[0, 0, ...]
)

preserve_dataset = YourDataset(
    images=[cat1.jpg, bird1.jpg, car1.jpg, ...],
    labels=[1, 2, 3, ...]
)

forget_dataloader = DataLoader(forget_dataset, batch_size=32)
preserve_dataloader = DataLoader(preserve_dataset, batch_size=32)
```

### 2. 中性目标选择
推荐使用 `method='random'`，因为：
- 最简单
- 不依赖模型前向传播
- 效果可能与噪声图像相当

### 3. 超参数
建议从文本侧的配置开始：
- ViT-B/16: `lamb_forget=1.1-1.3, lamb_preserve=0.25-0.3`
- RN50: `lamb_forget=1.3, lamb_preserve=0.4`

### 4. 验证
训练后检查：
```python
# 1. 遗忘效果
forget_similarity = model.encode_image(forget_images) @ model.encode_text(["dog"]).T
# 应该很低

# 2. 保留效果
preserve_similarity = model.encode_image(cat_images) @ model.encode_text(["cat"]).T
# 应该保持高

# 3. 文本-图像对齐（如果关心）
text_dog = model.encode_text(["dog"])
image_dog = model.encode_image(dog_images)
alignment = (text_dog @ image_dog.T).mean()
# 会降低（因为只修改了图像侧）
```

---

## 总结

| 方面 | 相同点 | 不同点 |
|------|--------|--------|
| **损失函数** | 完全相同的三组件损失 | 无 |
| **LoRA结构** | 相同的低秩适配方式 | 维度不同 |
| **训练流程** | 相同的优化循环 | 无 |
| **输入数据** | 无 | 文本 vs 图像 |
| **Hook层** | 都是编码器最后一层 | ln_final vs ln_post/attnpool |
| **中性目标** | 都需要定义 | 空字符串 vs 随机向量 |
| **架构处理** | 无 | 图像侧需要区分ViT/ResNet |
