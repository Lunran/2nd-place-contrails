import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .layers import FPN, UnetBlock, UpBlock


class ResNet18_U(nn.Module):
    """
    ResNet18ベースの単一フレーム用セグメンテーションモデル

    時系列モデリング(LSTM)を削除し、単純なU-Netアーキテクチャのみ使用。
    最もシンプルなベースラインとして、学習可能性を検証するため。

    パラメータ数: ~13M (LSTM削除でさらに軽量化)
    """

    def __init__(self, pretrained=True, num_classes=1, ps=0.0, weights_path=None, **kwargs):
        super().__init__()

        # ResNet18エンコーダー (事前学習済み)
        if pretrained and weights_path is None:
            # デフォルト: submit_modelディレクトリから読み込み
            weights_path = os.path.join(os.path.dirname(__file__), "..", "submit_model", "resnet18-imagenet.pth")

        if pretrained and weights_path and os.path.exists(weights_path):
            # ローカルファイルから重みを読み込み
            backbone = resnet18(weights=None)
            state_dict = torch.load(weights_path, map_location="cpu")
            backbone.load_state_dict(state_dict)
        else:
            # オンラインダウンロード（フォールバック）
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # ResNet18の各層を抽出
        self.enc0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool  # 64, stride=2  # stride=2
        )
        self.enc1 = backbone.layer1  # 64 channels, stride=1 (total: 4x down)
        self.enc2 = backbone.layer2  # 128 channels, stride=2 (total: 8x down)
        self.enc3 = backbone.layer3  # 256 channels, stride=2 (total: 16x down)
        self.enc4 = backbone.layer4  # 512 channels, stride=2 (total: 32x down)

        # チャンネル数 (ResNet18)
        nc = [64, 64, 128, 256, 512]

        # U-Netスタイルのデコーダー (LSTMなし)
        self.dec4 = UnetBlock(nc[-1], nc[-2], 384)  # 512+256 -> 384
        self.dec3 = UnetBlock(384, nc[-3], 192)  # 384+128 -> 192
        self.dec2 = UnetBlock(192, nc[-4], 96)  # 192+64 -> 96

        # Feature Pyramid Network
        self.fpn = FPN([nc[-1], 384, 192], [32] * 3)

        # Dropout
        self.drop = nn.Dropout2d(ps)

        # 最終出力層
        self.final_conv = UpBlock(96 + 32 * 3, num_classes, blur=True)

        self.up_result = 1  # 最終的な2倍アップサンプリング

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W) = (batch, 3, 5, 256, 256)
               または (B, C, H, W) = (batch, 3, 256, 256)
               - 3 channels: false color image (R/G/B)
               - 時系列の場合は最後のフレームのみ使用

        Returns:
            output: (B, num_classes, H*2, W*2) = (batch, 1, 256, 256)
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
        # enc0: (B, 64, 128, 128)   # 4x down
        # enc1: (B, 64, 128, 128)   # 4x down
        # enc2: (B, 128, 64, 64)    # 8x down
        # enc3: (B, 256, 32, 32)    # 16x down
        # enc4: (B, 512, 16, 16)    # 32x down
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # U-Netデコーダー
        dec4 = enc4  # (B, 512, 16, 16)
        dec3 = self.dec4(dec4, enc3)  # (B, 384, 32, 32)
        dec2 = self.dec3(dec3, enc2)  # (B, 192, 64, 64)
        dec1 = self.dec2(dec2, enc1)  # (B, 96, 128, 128)

        # FPN: マルチスケール特徴の融合
        x = self.fpn([dec4, dec3, dec2], dec1)  # (B, 96+32*3, 128, 128)

        # 最終出力
        x = self.final_conv(self.drop(x))  # (B, 1, 256, 256)

        # オプション: 追加のアップサンプリング
        if self.up_result != 0:
            x = F.interpolate(x, scale_factor=self.up_result, mode="bilinear")

        return x


if __name__ == "__main__":
    # テスト
    model = ResNet18_U(pretrained=True)

    # パラメータ数を計算
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ダミー入力でテスト (時系列)
    print("\n=== Test 1: Temporal input (B, C, T, H, W) ===")
    x = torch.randn(2, 3, 5, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 1, 256, 256), f"Expected (2, 1, 256, 256), got {y.shape}"

    # ダミー入力でテスト (単一フレーム)
    print("\n=== Test 2: Single frame input (B, C, H, W) ===")
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 1, 256, 256), f"Expected (2, 1, 256, 256), got {y.shape}"

    print("\n✓ All model tests passed!")
