#!/usr/bin/env python3
"""
ResNet18 ImageNet事前学習済み重みをダウンロードするスクリプト
"""
import sys
from pathlib import Path

import torch
from torchvision.models import ResNet18_Weights, resnet18

DOWNLOAD_DIR = "experiments"


def download_resnet18_weights(output_dir):
    """
    ResNet18のImageNet事前学習済み重みをダウンロード

    Args:
        output_dir: 保存先ディレクトリ
    """
    # 出力ディレクトリを作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存先のファイルパス
    weights_file = output_path / "resnet18-imagenet.pth"

    if weights_file.exists():
        print(f"✓ 重みファイルは既に存在します: {weights_file}")
        return str(weights_file)

    print("ResNet18のImageNet事前学習済み重みをダウンロード中...")

    # 事前学習済みモデルをロード (重みは自動的にキャッシュからダウンロードされる)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # state_dictを保存
    state_dict = model.state_dict()
    torch.save(state_dict, weights_file)

    print(f"✓ 重みを保存しました: {weights_file}")
    print(f"  ファイルサイズ: {weights_file.stat().st_size / 1024 / 1024:.2f} MB")

    return str(weights_file)


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else DOWNLOAD_DIR
    weights_file = download_resnet18_weights(output_dir)
