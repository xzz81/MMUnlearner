import os
import pytorch_lightning as pl
from .finetune_module import FinetuneLightningModule
from ..common.model_loader import load_model_and_processor
from ..dataloader.mllmu_datamodule import MLLMUDataModule


def run_finetune(config):
    """运行 finetune 阶段"""
    model, processor = load_model_and_processor(config)
    dm = MLLMUDataModule(config, processor)
    dm.setup()

    train_config = config.get("train", {})
    runtime_config = config.get("config", {}).get("runtime", {})
    output_config = config.get("config", {}).get("output", {})

    # 创建 Lightning Module
    module = FinetuneLightningModule(model, processor, config)

    # 创建 Trainer
    trainer = pl.Trainer(
        max_epochs=train_config.get("num_epochs", 5),
        accelerator=runtime_config.get("accelerator", "auto"),
        devices=runtime_config.get("devices", "auto"),
        strategy=runtime_config.get("strategy", "auto"),
        precision=runtime_config.get("precision", "bf16-mixed"),
        gradient_clip_val=train_config.get("gradient_clip", 1.0),
        accumulate_grad_batches=train_config.get("finetune", {}).get("gradient_accumulation_steps", 1),
        default_root_dir=output_config.get("base_dir", "output"),
    )

    # 训练
    trainer.fit(module, dm)

    # 保存模型
    save_dir = os.path.join(
        output_config.get("base_dir", "output"),
        output_config.get("exp_name") or "finetune"
    )
    os.makedirs(save_dir, exist_ok=True)

    # 合并 LoRA 并保存
    from peft import PeftModel
    if isinstance(module.model, PeftModel):
        merged_model = module.model.merge_and_unload()
        merged_model.save_pretrained(save_dir)
    else:
        module.model.save_pretrained(save_dir)

    print(f"Model saved to: {save_dir}")
    return module
