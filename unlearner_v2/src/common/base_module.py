import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import get_scheduler


class BaseLightningModule(pl.LightningModule):
    def __init__(self, model, processor, config):
        super().__init__()
        self.model = model
        self.processor = processor
        self.config = config
        self.save_hyperparameters(ignore=["model", "processor"])

    def configure_optimizers(self):
        train_config = self.config.get("train", {})
        lr = train_config.get("lr", 5e-4)
        weight_decay = train_config.get("optimizer", {}).get("weight_decay", 0.01)

        optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler_config = train_config.get("scheduler", {})
        scheduler_type = scheduler_config.get("type", "linear")
        warmup_steps = scheduler_config.get("warmup_steps", 0)

        scheduler = get_scheduler(
            scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
