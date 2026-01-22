import torch
from transformers import (
    LlavaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model


def find_all_linear_names(model):
    """查找模型中所有可训练的线性层名称"""
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model', 'visual']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def load_model_and_processor(config: dict):
    """根据配置加载模型和处理器"""
    model_config = config["model"]
    model_type = model_config["model_type"]
    model_path = model_config.get("vanilla_path") or model_config["model_path"]
    lora_config_dict = model_config.get("lora", {})

    # 加载模型 - 不使用 device_map，让 Lightning 管理设备
    if model_type == "llava":
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    elif model_type == "qwen3-vl":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    elif model_type == "qwen2-vl":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # 应用 LoRA
    if lora_config_dict.get("enable", False):
        target_modules = lora_config_dict.get("target_modules", "auto")
        if target_modules == "auto":
            target_modules = find_all_linear_names(model)

        lora_config = LoraConfig(
            r=lora_config_dict.get("r", 16),
            lora_alpha=lora_config_dict.get("alpha", 16),
            lora_dropout=lora_config_dict.get("dropout", 0.05),
            target_modules=target_modules,
            init_lora_weights="gaussian",
        )
        if model_type == "llava":
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # 加载处理器
    processor = AutoProcessor.from_pretrained(model_config["model_path"])
    processor.tokenizer.padding_side = "right"
    if model_type == "llava":
        processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)
        model.resize_token_embeddings(len(processor.tokenizer))

    return model, processor
