import torch
from torch import nn


class MergerUnlearningLoss(nn.Module):
    def __init__(self, lamb_forget=1.3, lamb_preserve=0.4, lamb_weight=1.0):
        super().__init__()
        self.lamb_forget = lamb_forget
        self.lamb_preserve = lamb_preserve
        self.lamb_weight = lamb_weight

    def forward(
        self,
        new_forget_emb,
        neutral_targets,
        new_retain_emb,
        original_retain_emb,
        lora_A,
        lora_B,
        lora_scale=1.0,
    ):
        forget_loss = self.lamb_forget * torch.norm(neutral_targets - new_forget_emb, p=2)
        preserve_loss = self.lamb_preserve * torch.norm(original_retain_emb - new_retain_emb, p=2)

        if lora_A is None or lora_B is None:
            weight_loss = torch.zeros((), device=new_forget_emb.device)
        else:
            weight_delta = (lora_B @ lora_A) * lora_scale
            weight_loss = self.lamb_weight * torch.norm(weight_delta.T, p=2)

        total_loss = forget_loss + preserve_loss + weight_loss
        return total_loss, {
            "forget": forget_loss,
            "preserve": preserve_loss,
            "weight": weight_loss,
        }


def compute_neutral_target(num_samples, output_dim, device):
    """Generate random normalized vectors as neutral targets."""
    targets = torch.randn(num_samples, output_dim, device=device)
    return targets / targets.norm(dim=-1, keepdim=True)
