import os
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from io import BytesIO
from PIL import Image
import json


class EvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        samples = []
        for _, row in self.df.iterrows():
            image_data = row['image'].get('bytes')
            try:
                image = Image.open(BytesIO(image_data)).convert("RGB")
            except Exception as e:
                print(f"Error loading image: {e}")
                continue

            if 'metadata' in row.index:
                try:
                    metadata = json.loads(row['metadata'])
                except json.JSONDecodeError:
                    continue
                for qa_pair in metadata:
                    samples.append({
                        "image": image,
                        "question": qa_pair.get("Question", ""),
                        "answer": qa_pair.get("Answer", "")
                    })
            else:
                samples.append({
                    "image": image,
                    "question": row.get('question', ""),
                    "answer": row.get('answer', "")
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class EvalDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.test_dataset = None

    def setup(self, stage=None):
        data_config = self.config["data"]
        eval_file = data_config.get("eval_file")

        if eval_file and os.path.exists(eval_file):
            df = pd.read_parquet(eval_file)
            self.test_dataset = EvalDataset(df)

    def predict_dataloader(self):
        data_config = self.config["data"]

        def collate_fn(batch):
            return {
                "images": [item["image"] for item in batch],
                "questions": [item["question"] for item in batch],
                "answers": [item["answer"] for item in batch]
            }

        return DataLoader(
            self.test_dataset,
            batch_size=data_config.get("batch_size", 4),
            shuffle=False,
            num_workers=data_config.get("num_workers", 4),
            collate_fn=collate_fn
        )
