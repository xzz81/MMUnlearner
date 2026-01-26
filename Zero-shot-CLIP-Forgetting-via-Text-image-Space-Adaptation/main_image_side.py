"""
图像侧遗忘实现 - 从文本侧类比过来

核心思路：
- 文本侧修改 text_projection
- 图像侧修改 visual.proj (对于ViT) 或 visual.attnpool.c_proj (对于ResNet)
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import clip
from utils_lora import Linear
import numpy as np

# ============================================================================
# 1. Hook机制 - 提取图像编码器中间层特征
# ============================================================================

hooks = {}

def get_activation_visual(name):
    """
    类比文本侧的 get_activation_proj

    文本侧: 提取 ln_final 的输出
    图像侧: 提取 ln_post 的输出 (ViT) 或 attnpool 的输出 (ResNet)
    """
    def hook(model, input, output):
        global hooks
        # ViT: ln_post 是最后的LayerNorm
        # ResNet: attnpool 是注意力池化层
        if name in ['ln_post', 'attnpool']:
            hooks[name] = output.detach()
    return hook


def register_visual_hooks(model):
    """
    类比文本侧的 register_model_hooks

    给图像编码器注册hook
    """
    # 清空已有的hooks
    for name, module in model.visual.named_modules():
        module._forward_hooks = OrderedDict()

    # 注册新的hooks
    for name, module in model.visual.named_modules():
        module.register_forward_hook(get_activation_visual(name))


# ============================================================================
# 2. 预计算图像特征 - 类比 precompute_projections
# ============================================================================

@torch.no_grad()
def precompute_image_features(model, image_dataloader, device):
    """
    类比文本侧的 precompute_projections

    文本侧输入: 类别名称列表 ["dog", "cat", ...]
    图像侧输入: 图像数据加载器 (包含实际图像)

    返回:
    - projections: 最终的图像嵌入 [N, 512]
    - list_hooks: ln_post/attnpool的输出特征 [N, hidden_dim]
    """
    global hooks
    list_hooks = []
    projections = []

    for batch in image_dataloader:
        images = batch['images'].to(device)
        hooks = {}

        # 前向传播，触发hook
        image_embeddings = model.encode_image(images)
        projections.append(image_embeddings)

        # 提取hook特征
        # ViT: ln_post输出 shape [batch, 768]
        # ResNet: attnpool输出 shape [batch, 2048]
        if 'ln_post' in hooks:
            list_hooks.append(hooks['ln_post'].clone())
        elif 'attnpool' in hooks:
            list_hooks.append(hooks['attnpool'].clone())

    projections = torch.cat(projections, dim=0)
    list_hooks = torch.cat(list_hooks, dim=0)

    return projections, list_hooks


# ============================================================================
# 3. 计算中性目标 - 类比 compute_proj_into
# ============================================================================

@torch.no_grad()
def compute_neutral_image_target(model, num_forget_classes, device, method='random'):
    """
    类比文本侧的 compute_proj_into

    文本侧: 使用空字符串 "" 的嵌入作为中性目标
    图像侧: 需要定义"中性"图像

    方法选项:
    1. 'random': 随机噪声向量（归一化）
    2. 'mean': 所有类别的平均嵌入
    3. 'noise_image': 噪声图像的嵌入
    """

    if method == 'random':
        # 方法1: 纯随机向量
        proj_into = torch.randn(num_forget_classes, 512).to(device)
        proj_into = proj_into / proj_into.norm(dim=-1, keepdim=True)

    elif method == 'noise_image':
        # 方法2: 噪声图像的嵌入
        # 类比: empty_text = clip.tokenize("")
        noise_image = torch.randn(num_forget_classes, 3, 224, 224).to(device)
        # 归一化到[0, 1]
        noise_image = (noise_image - noise_image.min()) / (noise_image.max() - noise_image.min())

        embed = model.encode_image(noise_image)
        proj_into = embed / embed.norm(dim=-1, keepdim=True)

    elif method == 'mean':
        # 方法3: 使用所有类别的平均嵌入（需要额外传入）
        # 这个需要在外部计算
        raise NotImplementedError("Use 'random' or 'noise_image' method")

    return proj_into


# ============================================================================
# 4. 主训练循环 - 类比 main.py 的训练部分
# ============================================================================

def train_image_side_forgetting(
    model,
    forget_dataloader,      # 要遗忘的类别的图像
    preserve_dataloader,    # 要保留的类别的图像
    device,
    lora_r=5,
    lamb_forget=1.3,
    lamb_preserve=0.4,
    lamb_weight=1.0,
    epochs=2000,
    lr=0.01
):
    """
    类比文本侧的训练循环 (main.py:284-330)

    主要修改:
    1. 输入从文本变为图像
    2. 修改的层从 text_projection 变为 visual.proj
    3. Hook从 ln_final 变为 ln_post
    """

    # 注册hook
    register_visual_hooks(model)

    # ========================================================================
    # 步骤1: 预计算特征
    # ========================================================================
    print("预计算遗忘类别的图像特征...")
    forget_projections, forget_hooks = precompute_image_features(
        model, forget_dataloader, device
    )

    print("预计算保留类别的图像特征...")
    preserve_projections, preserve_hooks = precompute_image_features(
        model, preserve_dataloader, device
    )

    # preserve_output: 保留类别的原始嵌入（目标）
    preserve_output = preserve_projections

    # ========================================================================
    # 步骤2: 计算中性目标
    # ========================================================================
    num_forget = forget_hooks.shape[0]
    proj_into = compute_neutral_image_target(model, num_forget, device, method='random')

    # ========================================================================
    # 步骤3: 创建LoRA层
    # ========================================================================
    # 获取visual projection的维度
    # ViT: visual.proj shape [768, 512]
    # ResNet: visual.attnpool.c_proj shape [2048, 512]

    if hasattr(model.visual, 'proj') and model.visual.proj is not None:
        # ViT架构
        visual_proj = model.visual.proj
        in_proj, out_proj = visual_proj.shape  # [768, 512]
        proj_type = 'vit'
    else:
        # ResNet架构
        visual_proj = model.visual.attnpool.c_proj.weight
        out_proj, in_proj = visual_proj.shape  # [512, 2048]
        proj_type = 'resnet'

    print(f"架构类型: {proj_type}, 输入维度: {in_proj}, 输出维度: {out_proj}")

    # 创建LoRA层
    # 类比: new_text_proj = Linear(in_proj, out_proj, r=r, bias=False, device=device)
    new_visual_proj = Linear(in_proj, out_proj, r=lora_r, bias=False, device=device)

    if proj_type == 'vit':
        # ViT: proj是Parameter，shape [768, 512]
        new_visual_proj.weight = nn.Parameter(visual_proj.T)  # 转置为 [512, 768]
    else:
        # ResNet: c_proj.weight shape [512, 2048]
        new_visual_proj.weight = nn.Parameter(visual_proj.T)  # 转置为 [2048, 512]

    new_visual_proj.weight.requires_grad = False

    # ========================================================================
    # 步骤4: 优化器
    # ========================================================================
    optimizer = torch.optim.Adam(list(new_visual_proj.parameters()), lr=lr)

    # ========================================================================
    # 步骤5: 训练循环
    # ========================================================================
    best_loss = np.inf
    best_weights = None

    with torch.no_grad():
        initial_forget_loss = torch.norm(proj_into - new_visual_proj(forget_hooks), p=2)

    print(f"初始遗忘损失: {initial_forget_loss.item():.4f}")

    for epoch in range(epochs):
        # 前向传播
        new_preserve_output = new_visual_proj(preserve_hooks)
        new_forget_output = new_visual_proj(forget_hooks)

        # 计算损失
        # 类比: loss = lamb_forget * torch.norm(proj_into - new_forget_output, p=2) + ...
        forget_loss = lamb_forget * torch.norm(proj_into - new_forget_output, p=2)
        preserve_loss = lamb_preserve * torch.norm(preserve_output - new_preserve_output, p=2)
        weight_loss = lamb_weight * torch.norm(
            (new_visual_proj.lora_B @ new_visual_proj.lora_A).transpose(0, 1), p=2
        )

        loss = forget_loss + preserve_loss + weight_loss

        # 打印
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: "
                  f"forget={forget_loss.item():.4f}, "
                  f"preserve={preserve_loss.item():.4f}, "
                  f"weight={weight_loss.item():.4f}, "
                  f"total={loss.item():.4f}")

        # 保存最佳权重
        if loss < best_loss:
            best_loss = loss
            new_visual_proj.train(False)
            best_weights = new_visual_proj.weight.clone()
            new_visual_proj.train(True)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # ========================================================================
    # 步骤6: 检查遗忘效果
    # ========================================================================
    with torch.no_grad():
        final_proj = forget_hooks @ best_weights.T
        final_forget_loss = torch.norm(proj_into - final_proj, p=2)
        reduction = (initial_forget_loss - final_forget_loss).abs() / initial_forget_loss

        print(f"\n最终遗忘损失: {final_forget_loss.item():.4f}")
        print(f"遗忘效果: {reduction.item():.2%}")

    # ========================================================================
    # 步骤7: 更新模型
    # ========================================================================
    if proj_type == 'vit':
        # ViT: 更新 visual.proj
        model.visual.proj = nn.Parameter(best_weights.T)
    else:
        # ResNet: 更新 visual.attnpool.c_proj.weight
        model.visual.attnpool.c_proj.weight = nn.Parameter(best_weights.T)

    return model


# ============================================================================
# 5. 使用示例
# ============================================================================

if __name__ == '__main__':
    """
    使用示例：

    文本侧输入: 类别名称
    图像侧输入: 实际图像数据
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载CLIP模型
    model, preprocess = clip.load("ViT-B/16", device=device)

    # 准备数据加载器（需要你自己实现）
    # forget_dataloader: 包含要遗忘类别的图像
    # preserve_dataloader: 包含要保留类别的图像

    # 示例数据加载器结构:
    # for batch in forget_dataloader:
    #     batch['images']: [batch_size, 3, 224, 224]
    #     batch['labels']: [batch_size]

    # 训练
    # model = train_image_side_forgetting(
    #     model=model,
    #     forget_dataloader=forget_dataloader,
    #     preserve_dataloader=preserve_dataloader,
    #     device=device,
    #     lora_r=5,
    #     lamb_forget=1.3,
    #     lamb_preserve=0.4,
    #     lamb_weight=1.0,
    #     epochs=2000,
    #     lr=0.01
    # )

    # 保存模型
    # torch.save(model.state_dict(), "model_image_forget.pth")

    print("图像侧遗忘实现完成！")
