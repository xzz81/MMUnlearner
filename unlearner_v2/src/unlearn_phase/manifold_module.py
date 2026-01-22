import torch
from ..common.base_module import BaseLightningModule


class ManifoldLightningModule(BaseLightningModule):
    def __init__(self, model, processor, config, grad_mask_path=None):
        super().__init__(model, processor, config)
        manifold_config = config.get("train", {}).get("unlearn", {}).get("manifold", {})
        self.forget_alpha = manifold_config.get("forget_alpha", 1.0)
        self.grad_mask = self._load_grad_mask(grad_mask_path) if grad_mask_path else None

    def _load_grad_mask(self, grad_mask_path):
        grad_data = torch.load(grad_mask_path)
        grad_mask = grad_data['weight']
        layer_name_list = list(grad_mask.keys())
        for name in layer_name_list:
            if any(kw in name for kw in ["proj", "fc", "linear", "mlp", "qkv"]):
                grad_mask.pop(name)

        total_cnt, w_cnt = 0, 0
        for k, v in grad_mask.items():
            num_zero = v.numel() - torch.count_nonzero(v)
            total_cnt += v.numel()
            w_cnt += num_zero
        print(f"Grad mask loaded. Sparsity: {w_cnt/total_cnt*100:.2f}%")
        return grad_mask

    def training_step(self, batch, batch_idx):
        batch_type = batch.pop("batch_type")
        outputs = self.model(**batch)

        if batch_type == "forget":
            loss = -self.forget_alpha * outputs.loss
            self.log("train/forget_loss", -loss.item())
        else:
            loss = outputs.loss
            self.log("train/retain_loss", loss.item())

        return loss

    def on_before_optimizer_step(self, optimizer):
        if self.grad_mask:
            for name, p in self.named_parameters():
                if p.grad is not None and name in self.grad_mask:
                    p.grad *= self.grad_mask[name].to(p.grad.device)
