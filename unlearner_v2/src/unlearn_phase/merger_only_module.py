import torch
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model

from ..common.base_module import BaseLightningModule
from ..losses import MergerUnlearningLoss, compute_neutral_target


class CachedMergerDataset(Dataset):
    def __init__(
        self,
        forget_features,
        forget_lengths,
        retain_features,
        retain_lengths,
        retain_original_embeddings,
        neutral_targets,
    ):
        self.forget_features = forget_features
        self.forget_lengths = forget_lengths
        self.retain_features = retain_features
        self.retain_lengths = retain_lengths
        self.retain_original_embeddings = retain_original_embeddings
        self.neutral_targets = neutral_targets
        self.num_forget = forget_features.shape[0]
        self.num_retain = retain_features.shape[0]
        self.length = max(self.num_forget, self.num_retain)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        forget_idx = idx % self.num_forget
        retain_idx = idx % self.num_retain
        return {
            "forget_feat": self.forget_features[forget_idx],
            "forget_len": self.forget_lengths[forget_idx],
            "retain_feat": self.retain_features[retain_idx],
            "retain_len": self.retain_lengths[retain_idx],
            "retain_orig": self.retain_original_embeddings[retain_idx],
            "neutral": self.neutral_targets[forget_idx],
        }


class MergerOnlyLightningModule(BaseLightningModule):
    def __init__(self, model, processor, config):
        super().__init__(model, processor, config)

        merger_config = config.get("train", {}).get("unlearn", {}).get("merger_only", {})
        self.loss_fn = MergerUnlearningLoss(
            lamb_forget=merger_config.get("lamb_forget", 1.3),
            lamb_preserve=merger_config.get("lamb_preserve", 0.4),
            lamb_weight=merger_config.get("lamb_weight", 1.0),
        )

        self.lora_r = merger_config.get("lora_r", 5)
        self.lora_alpha = merger_config.get("lora_alpha", 5)

        self.forget_dataloader = None
        self.retain_dataloader = None

        self.forget_vit_features = None
        self.forget_lengths = None
        self.retain_vit_features = None
        self.retain_lengths = None
        self.retain_original_embeddings = None
        self.neutral_targets = None

        self.cached_dataset = None
        self.cached_loader = None
        self.cache_ready = False

        self.hooks = {}
        self.hook_handles = []
        self.features_are_normed = True
        self.spatial_merge_size = None

        self.lora_A = None
        self.lora_B = None
        self.lora_scale = 1.0

    def on_fit_start(self):
        self._prepare_cached_data()

    def train_dataloader(self):
        self._prepare_cached_data()
        return self.cached_loader

    def _prepare_cached_data(self):
        if self.cache_ready:
            return
        if self.forget_dataloader is None or self.retain_dataloader is None:
            raise RuntimeError("forget_dataloader and retain_dataloader must be set before training.")

        self._register_post_patch_merge_hook()
        self._precompute_features()
        self._remove_hooks()
        self._setup_lora_merger()
        self._build_cached_dataset()
        self.model.train()
        self.cache_ready = True

    def _get_base_model(self):
        if hasattr(self.model, "get_base_model"):
            return self.model.get_base_model()
        return self.model

    def _get_merger(self):
        base_model = self._get_base_model()
        if not hasattr(base_model, "visual") or not hasattr(base_model.visual, "merger"):
            raise RuntimeError("Model does not expose visual.merger for merger-only unlearning.")
        return base_model.visual.merger

    def _register_post_patch_merge_hook(self):
        merger = self._get_merger()
        if not hasattr(merger, "linear_fc1"):
            raise RuntimeError("visual.merger.linear_fc1 not found for hook registration.")

        def linear_fc1_input_hook(module, inputs, output):
            if not inputs:
                return
            self.hooks["post_patch_merge_flat"] = inputs[0].detach()

        handle = merger.linear_fc1.register_forward_hook(linear_fc1_input_hook)
        self.hook_handles.append(handle)
        self.features_are_normed = True

    def _remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.hooks.clear()

    def _resolve_spatial_merge_size(self):
        base_model = self._get_base_model()
        if hasattr(base_model.visual, "spatial_merge_size"):
            return base_model.visual.spatial_merge_size
        if hasattr(base_model.visual, "merger") and hasattr(base_model.visual.merger, "spatial_merge_size"):
            return base_model.visual.merger.spatial_merge_size
        raise RuntimeError("spatial_merge_size not found on model.visual.")

    def _move_batch_to_device(self, batch):
        device = self.device
        moved = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                moved[k] = v.to(device)
            else:
                moved[k] = v
        return moved

    def _get_image_grid_thw(self, batch):
        grid = batch.get("image_grid_thw")
        if grid is None:
            grid = batch.get("grid_thw")
        if grid is None:
            raise KeyError("image_grid_thw not found in batch.")
        return grid

    def _compute_split_sizes(self, image_grid_thw):
        grid = image_grid_thw.detach().to("cpu")
        if grid.dim() == 1:
            grid = grid.unsqueeze(0)
        merge = self.spatial_merge_size
        if isinstance(merge, (tuple, list)):
            merge_area = int(merge[0]) * int(merge[1])
        else:
            merge_area = int(merge) * int(merge)
        split_sizes = (grid.prod(dim=-1) // merge_area).tolist()
        return [int(s) for s in split_sizes]

    def _pad_sequences(self, sequences):
        if len(sequences) == 0:
            raise ValueError("No sequences to pad.")
        lengths = torch.tensor([seq.shape[0] for seq in sequences], device=sequences[0].device, dtype=torch.long)
        max_len = int(lengths.max().item())
        feat_dim = sequences[0].shape[1]
        padded = sequences[0].new_zeros((len(sequences), max_len, feat_dim))
        for i, seq in enumerate(sequences):
            padded[i, : seq.shape[0]] = seq
        return padded, lengths

    def _pad_feature_list(self, features, max_len):
        if len(features) == 0:
            raise ValueError("No features to pad.")
        feat_dim = features[0].shape[1]
        padded = []
        for feat in features:
            pad_len = max_len - feat.shape[0]
            if pad_len > 0:
                pad = feat.new_zeros((pad_len, feat_dim))
                feat = torch.cat([feat, pad], dim=0)
            padded.append(feat)
        return torch.stack(padded, dim=0)

    def _masked_mean(self, seq, lengths):
        max_len = seq.size(1)
        device = seq.device
        lengths = lengths.to(device)
        lengths_clamped = lengths.clamp(min=1)
        mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths_clamped.unsqueeze(1)
        summed = (seq * mask.unsqueeze(-1)).sum(dim=1)
        return summed / lengths_clamped.unsqueeze(-1)

    def _run_merger(self, x):
        merger = self._get_merger()
        if not self.features_are_normed and hasattr(merger, "norm"):
            x = merger.norm(x)
        x = merger.linear_fc1(x)
        if hasattr(merger, "act_fn"):
            x = merger.act_fn(x)
        else:
            x = torch.nn.functional.gelu(x)
        x = merger.linear_fc2(x)
        return x

    @torch.no_grad()
    def _precompute_features(self):
        self.model.eval()
        self.spatial_merge_size = self._resolve_spatial_merge_size()
        merger = self._get_merger()
        expected_dim = merger.linear_fc1.in_features

        forget_features, forget_lengths = self._collect_features(
            self.forget_dataloader,
            expected_dim,
            collect_original=False,
        )
        retain_features, retain_lengths, retain_original = self._collect_features(
            self.retain_dataloader,
            expected_dim,
            collect_original=True,
        )

        output_dim = merger.linear_fc2.out_features
        neutral_targets = compute_neutral_target(len(forget_features), output_dim, device="cpu")
        if retain_original:
            neutral_targets = neutral_targets.to(dtype=retain_original[0].dtype)

        self.forget_vit_features = forget_features
        self.forget_lengths = forget_lengths
        self.retain_vit_features = retain_features
        self.retain_lengths = retain_lengths
        self.retain_original_embeddings = retain_original
        self.neutral_targets = neutral_targets

        self.model.train()

    def _collect_features(self, dataloader, expected_dim, collect_original):
        features = []
        lengths = []
        original_embeddings = []

        for batch in dataloader:
            batch = self._move_batch_to_device(batch)
            if "labels" in batch:
                batch.pop("labels")
            self.hooks["post_patch_merge_flat"] = None
            _ = self.model(**batch)
            flat = self.hooks.get("post_patch_merge_flat")
            if flat is None:
                raise RuntimeError("Post-merge hook did not capture features.")
            if flat.shape[-1] != expected_dim:
                raise ValueError(
                    f"Merger input dim mismatch: expected {expected_dim}, got {flat.shape[-1]}."
                )

            image_grid_thw = self._get_image_grid_thw(batch)
            split_sizes = self._compute_split_sizes(image_grid_thw)
            if sum(split_sizes) != flat.shape[0]:
                if flat.dim() == 2:
                    raise ValueError("Split sizes do not match merged token count.")

            if flat.dim() == 3:
                if len(split_sizes) != flat.shape[0]:
                    raise ValueError("Batch size mismatch with image_grid_thw.")
                if max(split_sizes) > flat.shape[1]:
                    raise ValueError("Split sizes exceed sequence length.")
                split_features = [
                    flat[i, : split_sizes[i]]
                    for i in range(len(split_sizes))
                ]
            else:
                split_features = list(torch.split(flat, split_sizes, dim=0))
            features.extend([feat.detach().cpu() for feat in split_features])
            lengths.extend([feat.shape[0] for feat in split_features])

            if collect_original:
                padded, seq_lengths = self._pad_sequences(split_features)
                merged = self._run_merger(padded)
                pooled = self._masked_mean(merged, seq_lengths)
                for emb in pooled:
                    original_embeddings.append(emb.detach().cpu())

        if collect_original:
            return features, lengths, original_embeddings
        return features, lengths

    def _setup_lora_merger(self):
        if hasattr(self.model, "peft_config"):
            self.model = self.model.merge_and_unload()

        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["merger.linear_fc2"],
            lora_dropout=0.0,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )

        for param in self.model.parameters():
            param.requires_grad = False

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        self.lora_scale = self.lora_alpha / self.lora_r if self.lora_r else 1.0
        self._cache_lora_params()

    def _cache_lora_params(self):
        self.lora_A = None
        self.lora_B = None
        for name, param in self.model.named_parameters():
            if "merger.linear_fc2" in name and "lora_A" in name:
                self.lora_A = param
            if "merger.linear_fc2" in name and "lora_B" in name:
                self.lora_B = param
        if self.lora_A is None or self.lora_B is None:
            raise RuntimeError(
                "LoRA parameters for merger.linear_fc2 not found. "
                "Check target_modules or PEFT version."
            )

    def _build_cached_dataset(self):
        forget_max_len = max(self.forget_lengths)
        retain_max_len = max(self.retain_lengths)

        forget_features = self._pad_feature_list(self.forget_vit_features, forget_max_len)
        retain_features = self._pad_feature_list(self.retain_vit_features, retain_max_len)

        forget_lengths = torch.tensor(self.forget_lengths, dtype=torch.long)
        retain_lengths = torch.tensor(self.retain_lengths, dtype=torch.long)
        retain_original = torch.stack(self.retain_original_embeddings, dim=0)

        self.cached_dataset = CachedMergerDataset(
            forget_features=forget_features,
            forget_lengths=forget_lengths,
            retain_features=retain_features,
            retain_lengths=retain_lengths,
            retain_original_embeddings=retain_original,
            neutral_targets=self.neutral_targets,
        )

        batch_size = self.config.get("data", {}).get("batch_size", 4)
        self.cached_loader = DataLoader(
            self.cached_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

    def training_step(self, batch, batch_idx):
        device = self.device
        forget_feat = batch["forget_feat"].to(device)
        forget_len = batch["forget_len"].to(device)
        retain_feat = batch["retain_feat"].to(device)
        retain_len = batch["retain_len"].to(device)
        retain_orig = batch["retain_orig"].to(device)
        neutral = batch["neutral"].to(device)

        new_forget_seq = self._run_merger(forget_feat)
        new_retain_seq = self._run_merger(retain_feat)

        new_forget_emb = self._masked_mean(new_forget_seq, forget_len)
        new_retain_emb = self._masked_mean(new_retain_seq, retain_len)

        loss, loss_dict = self.loss_fn(
            new_forget_emb,
            neutral,
            new_retain_emb,
            retain_orig,
            self.lora_A,
            self.lora_B,
            self.lora_scale,
        )

        self.log("train/total_loss", loss)
        self.log("train/forget_loss", loss_dict["forget"])
        self.log("train/preserve_loss", loss_dict["preserve"])
        self.log("train/weight_loss", loss_dict["weight"])

        return loss

    def configure_optimizers(self):
        train_config = self.config.get("train", {})
        merger_config = train_config.get("unlearn", {}).get("merger_only", {})
        lr = merger_config.get("lr", 0.01)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable_params, lr=lr)

    def on_train_end(self):
        if hasattr(self.model, "merge_and_unload"):
            self.model = self.model.merge_and_unload()

    def teardown(self, stage: str):
        self._remove_hooks()
