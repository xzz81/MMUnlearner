"""
下载 MLLMU-Bench 和 CLEAR 数据集
支持从 HuggingFace 或镜像下载
"""

import os
import argparse

from huggingface_hub import snapshot_download, HfApi


def download_mllmu_bench(save_dir: str = "data/MLLMU-Bench"):
    """
    下载 MLLMU-Bench 数据集
    https://huggingface.co/datasets/MLLMMU/MLLMU-Bench
    """
    print("=" * 50)
    print("正在下载 MLLMU-Bench 数据集...")
    print("=" * 50)

    os.makedirs(save_dir, exist_ok=True)

    try:
        dataset_path = snapshot_download(
            repo_id='MLLMMU/MLLMU-Bench',
            repo_type='dataset',
            local_dir=save_dir,
        )
        print(f"MLLMU-Bench 下载完成，保存在: {dataset_path}")
        return dataset_path
    except Exception as e:
        print(f"下载失败: {e}")
        raise


def download_clear(save_dir: str = "data/CLEAR"):
    """
    下载 CLEAR 数据集
    https://huggingface.co/datasets/therem/CLEAR
    """
    print("=" * 50)
    print("正在下载 CLEAR 数据集...")
    print("=" * 50)

    os.makedirs(save_dir, exist_ok=True)

    try:
        dataset_path = snapshot_download(
            repo_id='therem/CLEAR',
            repo_type='dataset',
            local_dir=save_dir,
        )
        print(f"CLEAR 下载完成，保存在: {dataset_path}")
        return dataset_path
    except Exception as e:
        print(f"下载失败: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='下载 MLLMU-Bench 和 CLEAR 数据集')
    parser.add_argument('--mirror', type=str, default=None,
                        help='HuggingFace 镜像地址, 如: https://hf-mirror.com')
    parser.add_argument('--only', type=str, default=None,
                        choices=['mllmu', 'clear'],
                        help='只下载指定数据集')
    args = parser.parse_args()

    # 设置镜像
    if args.mirror:
        os.environ['HF_ENDPOINT'] = args.mirror
        print(f"使用镜像: {args.mirror}")

    print("\n" + "=" * 60)
    print("开始下载数据集")
    print("=" * 60 + "\n")

    # 确保 data 目录存在
    os.makedirs("data", exist_ok=True)

    # 下载 MLLMU-Bench
    if args.only is None or args.only == 'mllmu':
        try:
            mllmu_path = download_mllmu_bench("data/MLLMU-Bench")
            print(f"\n[OK] MLLMU-Bench 数据集已保存到: {mllmu_path}\n")
        except Exception as e:
            print(f"\n[FAIL] MLLMU-Bench 下载失败: {e}\n")

    # 下载 CLEAR
    if args.only is None or args.only == 'clear':
        try:
            clear_path = download_clear("data/CLEAR")
            print(f"\n[OK] CLEAR 数据集已保存到: {clear_path}\n")
        except Exception as e:
            print(f"\n[FAIL] CLEAR 下载失败: {e}\n")

    print("\n" + "=" * 60)
    print("数据集下载完成!")
    print("=" * 60)
    print("\n请检查以下目录:")
    print("  - data/MLLMU-Bench")
    print("  - data/CLEAR")


if __name__ == "__main__":
    main()
