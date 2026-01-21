import os
import sys
import argparse

import pandas as pd
from torch.utils.data import DataLoader, random_split
sys.path.append(('.'))
sys.path.append(('../'))
sys.path.append(('../../'))
from data_process.MLLMU_process import K_TYPE, P_TYPE, MLLMU_manifold_Dataset,train_collate_fn_llava_new,MLLMU_text_Dataset
from data_process.CLEAR_process import train_collate_clear
from data_process.data_preprocess import  LLAVA_multimodal_Dataset
import torch
import os
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, Qwen2VLForConditionalGeneration, Qwen3VLForConditionalGeneration
import torch
from SFRon import Mask_Our, Mask_grad


def main(args):
    model_id = args.model_id
    model_path = args.model_id

    # Load model based on model type
    if "qwen3" in model_id.lower():
        print("Loading Qwen3-VL model...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
    elif "qwen" in model_id.lower():
        print("Loading Qwen2-VL model...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
        )
    elif "llava" in model_id.lower():
        print("Loading LLaVA model...")
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
        )
    else:
        raise ValueError(f"Unsupported model: {model_id}")

    model.gradient_checkpointing_enable()

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right"
    print(model)

    mg = Mask_Our(model, 1e-5)

    forget_ratio = args.forget_ratio
    data_dir = args.data_dir
    forget_data_folder = os.path.join(data_dir, f"forget_{forget_ratio}")
    remain_data_folder = os.path.join(data_dir, f"retain_{100-forget_ratio}")

    # Define paths to the Parquet files for "forget" and "retain" datasets
    forget_parquet_file = os.path.join(forget_data_folder, "train-00000-of-00001.parquet")
    remain_parquet_file = os.path.join(remain_data_folder, "train-00000-of-00001.parquet")

    # Load DataLoader
    forget_df = pd.read_parquet(forget_parquet_file)
    retain_df = pd.read_parquet(remain_parquet_file)
    full_df = pd.read_parquet(os.path.join(data_dir, "Full_Set/train-00000-of-00001.parquet"))

    multimodal_forget_dataset = LLAVA_multimodal_Dataset(df=forget_df)
    multimodal_knowledge_dataset = MLLMU_text_Dataset(df=full_df)
    multimodal_remain_dataset = LLAVA_multimodal_Dataset(df=retain_df)

    language_preserve_dataset = torch.utils.data.ConcatDataset([multimodal_knowledge_dataset, multimodal_remain_dataset])
    vision_preserve_dataset = torch.utils.data.ConcatDataset([multimodal_remain_dataset])

    forget_dataloader = DataLoader(
        multimodal_forget_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda x: train_collate_clear(x, processor, model.device, True)
    )
    vision_preserve_dataloader = DataLoader(
        vision_preserve_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda x: train_collate_clear(x, processor, model.device, True)
    )
    language_preserve_dataloader = DataLoader(
        language_preserve_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda x: train_collate_clear(x, processor, model.device, True)
    )

    label = "ours"
    folder = args.output_dir
    os.makedirs(folder, exist_ok=True)

    weight_mask, forget_grad, preserve_grad = mg.prepare_weight_saliency_mask(
        modules=["visual"],
        forget_loader=forget_dataloader,
        preserve_loader=vision_preserve_dataloader,
        threshold=1,
        save_path=""
    )
    res = {"weight": weight_mask, "forget_grad": forget_grad, "preserve_grad": preserve_grad}
    torch.save(res, f"{folder}/mllmu_vision_mask_{label}.pt")

    weight_mask, forget_grad, preserve_grad = mg.prepare_weight_saliency_mask(
        modules=["language_model"],
        forget_loader=forget_dataloader,
        preserve_loader=language_preserve_dataloader,
        threshold=1,
        save_path=""
    )
    res = {"weight": weight_mask, "forget_grad": forget_grad, "preserve_grad": preserve_grad}
    torch.save(res, f"{folder}/mllmu_language_mask_{label}.pt")

    weight_mask, forget_grad, preserve_grad = mg.get_weight_saliency_mask()
    res = {"weight": weight_mask, "forget_grad": forget_grad, "preserve_grad": preserve_grad}
    torch.save(res, f"{folder}/mllmu_both_mask_{label}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate gradient mask for MLLMU")
    parser.add_argument("--model_id", type=str, required=True, help="Path to the model")
    parser.add_argument("--data_dir", type=str, default="data/MLLMU-Bench", help="Path to MLLMU-Bench data")
    parser.add_argument("--forget_ratio", type=int, default=5, help="Forget ratio (5, 10, or 15)")
    parser.add_argument("--output_dir", type=str, default="output/mask/forget5", help="Output directory for masks")

    args = parser.parse_args()
    main(args)
