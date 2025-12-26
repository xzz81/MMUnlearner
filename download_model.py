"""
使用 ModelScope 下载 Qwen2-VL 模型
"""

import os
import argparse
from modelscope import snapshot_download


def download_qwen2_vl(model_id: str, save_dir: str):
    """
    下载 Qwen2-VL 模型

    Args:
        model_id: ModelScope 模型 ID
        save_dir: 保存目录
    """
    print("=" * 60)
    print(f"正在下载模型: {model_id}")
    print(f"保存目录: {save_dir}")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)

    try:
        model_path = snapshot_download(
            model_id=model_id,
            cache_dir=save_dir,
            revision='master'
        )
        print(f"\n[OK] 模型下载完成!")
        print(f"模型路径: {model_path}")
        return model_path
    except Exception as e:
        print(f"\n[FAIL] 下载失败: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='使用 ModelScope 下载 Qwen2-VL 模型')
    parser.add_argument('--model', type=str, default='7B',
                        choices=['2B', '7B', '72B'],
                        help='模型大小: 2B, 7B, 72B (默认: 7B)')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='保存目录 (默认: models)')
    args = parser.parse_args()

    # ModelScope 上 Qwen2-VL 的模型 ID
    model_map = {
        '2B': 'Qwen/Qwen2-VL-2B-Instruct',
        '7B': 'Qwen/Qwen2-VL-7B-Instruct',
        '72B': 'Qwen/Qwen2-VL-72B-Instruct',
    }

    model_id = model_map[args.model]

    print("\n" + "=" * 60)
    print("Qwen2-VL 模型下载器 (ModelScope)")
    print("=" * 60)
    print(f"选择模型: Qwen2-VL-{args.model}-Instruct")
    print(f"ModelScope ID: {model_id}")
    print("=" * 60 + "\n")

    model_path = download_qwen2_vl(model_id, args.save_dir)

    print("\n" + "=" * 60)
    print("下载完成!")
    print("=" * 60)
    print(f"\n使用示例:")
    print(f"  python MLLMU_finetune.py --model_id {model_path} ...")
    print(f"  python CLEAR_finetune.py --model_id {model_path} ...")


if __name__ == "__main__":
    main()
