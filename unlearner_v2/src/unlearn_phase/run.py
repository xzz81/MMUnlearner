import os
import json
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from itertools import cycle

from ..common.model_loader import load_model_and_processor
from ..dataloader.mllmu_datamodule import MLLMUDataModule
from .manifold_module import ManifoldLightningModule


class UnlearnDataLoader:
    """交替返回 forget 和 retain batch"""
    def __init__(self, forget_loader, retain_loader, forget_freq):
        self.forget_loader = forget_loader
        self.retain_loader = retain_loader
        self.forget_freq = forget_freq

    def __iter__(self):
        forget_iter = cycle(iter(self.forget_loader))
        for i, retain_batch in enumerate(iter(self.retain_loader)):
            if i % self.forget_freq == 0:
                forget_batch = next(forget_iter)
                forget_batch["batch_type"] = "forget"
                yield forget_batch
            retain_batch["batch_type"] = "retain"
            yield retain_batch

    def __len__(self):
        return len(self.retain_loader) + len(self.retain_loader) // self.forget_freq


def run_unlearn(config):
    model, processor = load_model_and_processor(config)

    manifold_config = config.get("train", {}).get("unlearn", {}).get("manifold", {})
    grad_mask_path = manifold_config.get("grad_mask_path")

    lightning_module = ManifoldLightningModule(
        model=model,
        processor=processor,
        config=config,
        grad_mask_path=grad_mask_path
    )

    data_module = MLLMUDataModule(config, processor)
    data_module.setup()

    forget_loader, retain_loader = data_module.unlearn_dataloaders()
    forget_freq = len(retain_loader) // len(forget_loader)
    combined_loader = UnlearnDataLoader(forget_loader, retain_loader, forget_freq)

    train_config = config.get("train", {})
    runtime_config = config.get("config", {}).get("runtime", {})

    checkpoint_callback = ModelCheckpoint(
        dirpath=train_config.get("checkpoint_dir", "./checkpoints"),
        filename="unlearn-{epoch:02d}",
        save_top_k=1,
        monitor="train/retain_loss"
    )

    trainer = pl.Trainer(
        max_epochs=train_config.get("num_epochs", 5),
        accelerator=runtime_config.get("accelerator", "auto"),
        devices=runtime_config.get("devices", "auto"),
        strategy=runtime_config.get("strategy", "auto"),
        precision=runtime_config.get("precision", "bf16-mixed"),
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
        log_every_n_steps=10
    )

    trainer.fit(lightning_module, combined_loader)

    save_dir = train_config.get("save_dir", "./saved_model")
    os.makedirs(save_dir, exist_ok=True)

    unwrapped_model = lightning_module.model
    if hasattr(unwrapped_model, 'merge_and_unload'):
        unwrapped_model = unwrapped_model.merge_and_unload()

    unwrapped_model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Model saved to: {save_dir}")


def run_merger_only_unlearn(config):
    """Run merger-only unlearning with precomputed features."""
    from .merger_only_module import MergerOnlyLightningModule

    model, processor = load_model_and_processor(config)
    data_module = MLLMUDataModule(config, processor)
    data_module.setup()

    forget_loader, retain_loader = data_module.unlearn_dataloaders()

    lightning_module = MergerOnlyLightningModule(
        model=model,
        processor=processor,
        config=config,
    )
    lightning_module.forget_dataloader = forget_loader
    lightning_module.retain_dataloader = retain_loader

    train_config = config.get("train", {})
    merger_config = train_config.get("unlearn", {}).get("merger_only", {})
    runtime_config = config.get("config", {}).get("runtime", {})

    checkpoint_callback = ModelCheckpoint(
        dirpath=train_config.get("checkpoint_dir", "./checkpoints"),
        filename="merger-only-{epoch:02d}",
        save_top_k=1,
        monitor="train/total_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=merger_config.get("num_epochs", 2000),
        accelerator=runtime_config.get("accelerator", "auto"),
        devices=runtime_config.get("devices", "auto"),
        strategy=runtime_config.get("strategy", "auto"),
        precision=runtime_config.get("precision", "bf16-mixed"),
        callbacks=[checkpoint_callback],
        log_every_n_steps=100,
        enable_progress_bar=True,
    )

    trainer.fit(lightning_module)

    save_dir = train_config.get("save_dir", "./output")
    os.makedirs(save_dir, exist_ok=True)

    unwrapped_model = lightning_module.model
    if hasattr(unwrapped_model, "merge_and_unload"):
        unwrapped_model = unwrapped_model.merge_and_unload()

    unwrapped_model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Model saved to: {save_dir}")
