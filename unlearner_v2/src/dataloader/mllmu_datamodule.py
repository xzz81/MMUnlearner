import os
import json
import pandas as pd
import pytorch_lightning as pl
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from ..common.collate_fn import train_collate_fn, train_collate_fn_ansonly


class MLLMUDataset(Dataset):
    """MLLMU 数据集，从 Parquet 文件加载"""

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df
        self.dataset = self._flatten_dataset()

    def _flatten_dataset(self):
        flattened_data = []
        for idx, row in self.df.iterrows():
            image_data = row['image'].get('bytes')
            try:
                image = Image.open(BytesIO(image_data)).convert("RGB")
            except Exception as e:
                print(f"Error loading image at index {idx}: {e}")
                continue

            if 'metadata' in row.index:
                try:
                    metadata = json.loads(row['metadata'])
                except json.JSONDecodeError:
                    continue
                for qa_pair in metadata:
                    question = qa_pair.get("Question", "")
                    answer = qa_pair.get("Answer", "")
                    if question and answer:
                        flattened_data.append({
                            "image": image,
                            "question": question,
                            "answer": answer
                        })
            else:
                question = row.get('question', "")
                answer = row.get('answer', "")
                if question and answer:
                    flattened_data.append({
                        "image": image,
                        "question": question,
                        "answer": answer
                    })
        return flattened_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class MLLMUDataModule(pl.LightningDataModule):
    def __init__(self, config, processor):
        super().__init__()
        self.config = config
        self.processor = processor
        self.train_dataset = None
        self.forget_dataset = None
        self.retain_dataset = None

    def setup(self, stage=None):
        data_config = self.config["data"]
        data_dir = data_config["data_dir"]

        # Finetune: 加载 Full_Set
        full_parquet = os.path.join(data_dir, "Full_Set", "train-00000-of-00001.parquet")
        if os.path.exists(full_parquet):
            df = pd.read_parquet(full_parquet)
            self.train_dataset = MLLMUDataset(df)

        # Unlearn: 加载 forget/retain
        forget_ratio = data_config.get("forget_split_ratio", 5)
        forget_folder = os.path.join(data_dir, f"forget_{forget_ratio}")
        retain_folder = os.path.join(data_dir, f"retain_{100 - forget_ratio}")

        forget_parquet = os.path.join(forget_folder, "train-00000-of-00001.parquet")
        retain_parquet = os.path.join(retain_folder, "train-00000-of-00001.parquet")

        if os.path.exists(forget_parquet):
            self.forget_dataset = MLLMUDataset(pd.read_parquet(forget_parquet))
        if os.path.exists(retain_parquet):
            self.retain_dataset = MLLMUDataset(pd.read_parquet(retain_parquet))

    def train_dataloader(self):
        data_config = self.config["data"]
        train_config = self.config.get("train", {})
        ans_only = train_config.get("ans_only", False)

        collate_fn = train_collate_fn_ansonly if ans_only else train_collate_fn

        return DataLoader(
            self.train_dataset,
            batch_size=data_config.get("batch_size", 4),
            shuffle=True,
            num_workers=0,
            collate_fn=lambda x: collate_fn(x, self.processor, None, True)
        )

    def unlearn_dataloaders(self):
        """返回 forget 和 retain 两个 dataloader"""
        data_config = self.config["data"]
        train_config = self.config.get("train", {})
        ans_only = train_config.get("ans_only", False)
        collate_fn = train_collate_fn_ansonly if ans_only else train_collate_fn

        forget_loader = DataLoader(
            self.forget_dataset,
            batch_size=data_config.get("batch_size", 4),
            shuffle=True,
            num_workers=0,
            collate_fn=lambda x: collate_fn(x, self.processor, None, True)
        )

        retain_loader = DataLoader(
            self.retain_dataset,
            batch_size=data_config.get("batch_size", 4),
            shuffle=True,
            num_workers=0,
            collate_fn=lambda x: collate_fn(x, self.processor, None, True)
        )

        return forget_loader, retain_loader
