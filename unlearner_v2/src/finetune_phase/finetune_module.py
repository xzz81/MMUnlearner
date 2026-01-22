from ..common.base_module import BaseLightningModule


class FinetuneLightningModule(BaseLightningModule):
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        self.log("val_loss", outputs.loss)
