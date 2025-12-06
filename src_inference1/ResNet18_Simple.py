import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18_Simple(nn.Module):
    """
    ResNet18ベースの最もシンプルなセグメンテーションモデル

    - エンコーダーのみ (U-Netスキップ接続なし)
    - デコーダーは単純な畳み込み層のみ
    - LSTM、FPN、UnetBlockなし

    最小限の構造で学習可能性を検証するためのベースライン。

    パラメータ数: ~12M (最も軽量)
    """

    def __init__(self, weights_path=None, num_classes=1, **kwargs):
        super().__init__()

        if weights_path:
            # ローカルファイルから重みを読み込み
            backbone = resnet18(weights=None)
            state_dict = torch.load(weights_path, map_location="cpu")
            backbone.load_state_dict(state_dict)
        else:
            # オンラインダウンロード（フォールバック）
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # ResNet18の各層を抽出
        self.enc0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool  # 64, stride=2  # stride=2
        )
        self.enc1 = backbone.layer1  # 64 channels, stride=1 (total: 4x down)
        self.enc2 = backbone.layer2  # 128 channels, stride=2 (total: 8x down)
        self.enc3 = backbone.layer3  # 256 channels, stride=2 (total: 16x down)
        self.enc4 = backbone.layer4  # 512 channels, stride=2 (total: 32x down)

        # シンプルなデコーダー (単純なアップサンプリング + 畳み込み)
        # 512 -> 256 -> 128 -> 64 -> num_classes
        self.decoder = nn.Sequential(
            # 16x16, 512 -> 32x32, 256
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 32x32, 256 -> 64x64, 128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 64x64, 128 -> 128x128, 64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 128x128, 64 -> 256x256, 32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 256x256, 32 -> 256x256, num_classes
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W) = (batch, 3, 5, 256, 256)
               または (B, C, H, W) = (batch, 3, 256, 256)
               - 3 channels: false color image (R/G/B)
               - 時系列の場合は最後のフレームのみ使用

        Returns:
            output: (B, num_classes, H, W) = (batch, 1, 256, 256)
        """
        # 時系列データの場合は最後のフレームのみ使用
        if x.dim() == 5:  # (B, C, T, H, W)
            x = x[:, :, -1]  # 最後のフレーム: (B, C, H, W)

        # 256x256 -> 512x512 (bicubic interpolation + clip to [0,1])
        x = F.interpolate(x, scale_factor=2, mode="bicubic").clip(0, 1)

        # ImageNet正規化 (ResNet18が期待する入力)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # エンコーダー (ResNet18各層)
        x = self.enc0(x)  # (B, 64, 128, 128)
        x = self.enc1(x)  # (B, 64, 128, 128)
        x = self.enc2(x)  # (B, 128, 64, 64)
        x = self.enc3(x)  # (B, 256, 32, 32)
        x = self.enc4(x)  # (B, 512, 16, 16)

        # デコーダー (単純なアップサンプリング)
        x = self.decoder(x)  # (B, 1, 256, 256)

        return x


class ResNet18_SimplerFCN(nn.Module):
    """
    さらにシンプルなFCNスタイルモデル

    - バイリニア補間によるアップサンプリング (転置畳み込みなし)
    - 最小限の計算コスト

    パラメータ数: ~11.5M
    """

    def __init__(self, weights_path=None, num_classes=1, **kwargs):
        super().__init__()

        if weights_path:
            # ローカルファイルから重みを読み込み
            backbone = resnet18(weights=None)
            state_dict = torch.load(weights_path, map_location="cpu")
            backbone.load_state_dict(state_dict)
        else:
            # オンラインダウンロード（フォールバック）
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # ResNet18の各層を抽出
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4

        # シンプルな1x1畳み込みでチャンネル削減
        self.reduce = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 最終出力層
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W) or (B, C, H, W)

        Returns:
            output: (B, num_classes, H, W)
        """
        # 時系列データの場合は最後のフレームのみ使用
        if x.dim() == 5:
            x = x[:, :, -1]

        # 元の入力サイズを保存
        _, _, h, w = x.shape

        # 512x512にアップサンプリング
        x = F.interpolate(x, scale_factor=2, mode="bicubic").clip(0, 1)

        # ImageNet正規化
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # エンコーダー
        x = self.enc0(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)  # (B, 512, 16, 16)

        # チャンネル削減
        x = self.reduce(x)  # (B, 32, 16, 16)

        # バイリニア補間で元のサイズに戻す
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

        # 最終出力
        x = self.final(x)  # (B, 1, 256, 256)

        return x


if __name__ == "__main__":
    print("=" * 60)
    print("ResNet18_Simple Test")
    print("=" * 60)

    model = ResNet18_Simple()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 時系列入力
    print("\n--- Test 1: Temporal input (B, C, T, H, W) ---")
    x = torch.randn(2, 3, 5, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 1, 256, 256)

    # 単一フレーム
    print("\n--- Test 2: Single frame (B, C, H, W) ---")
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 1, 256, 256)

    print("\n" + "=" * 60)
    print("ResNet18_SimplerFCN Test")
    print("=" * 60)

    model2 = ResNet18_SimplerFCN()

    total_params = sum(p.numel() for p in model2.parameters())
    trainable_params = sum(p.numel() for p in model2.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    x = torch.randn(2, 3, 5, 256, 256)
    y = model2(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 1, 256, 256)

    print("\n✓ All tests passed!")
