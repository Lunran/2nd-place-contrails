import torch
import torch.nn as nn
import torch.nn.functional as F

from .coat import coat_lite_medium, coat_lite_mini, coat_lite_small


class CoaT_SimplerFCN(nn.Module):
    """
    CoaTエンコーダーベースのシンプルなFCNスタイルモデル

    ResNet18_SimplerFCNのCoaT版:
    - CoaT-Liteをエンコーダーとして使用
    - バイリニア補間によるアップサンプリング (転置畳み込みなし)
    - 最小限の計算コスト
    - U-Netスキップ接続なし、LSTM/FPN/UnetBlockなし

    アーキテクチャ: mini, small, medium から選択可能
    """

    def __init__(self, pre=None, arch="medium", num_classes=1, **kwargs):
        super().__init__()

        # CoaT-Liteエンコーダーの選択
        if arch == "mini":
            self.enc = coat_lite_mini(return_interm_layers=True)
            nc = [64, 128, 320, 512]  # 各ステージのチャンネル数
        elif arch == "small":
            self.enc = coat_lite_small(return_interm_layers=True)
            nc = [64, 128, 320, 512]
        elif arch == "medium":
            self.enc = coat_lite_medium(return_interm_layers=True)
            nc = [128, 256, 320, 512]
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        # 事前学習済み重みの読み込み（オプション）
        if pre is not None:
            sd = torch.load(pre, map_location="cpu", weights_only=False)
            if "model" in sd:
                sd = sd["model"]
            print(self.enc.load_state_dict(sd, strict=False))

        # シンプルな1x1畳み込みでチャンネル削減
        # 最終ステージ(nc[-1])の特徴量のみを使用
        self.reduce = nn.Sequential(
            nn.Conv2d(nc[-1], 128, kernel_size=1),
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
               - 時系列データの場合は最後のフレームのみ使用

        Returns:
            output: (B, num_classes, H, W) = (batch, 1, 256, 256)
        """
        # 時系列データの場合は最後のフレームのみ使用
        if x.dim() == 5:  # (B, C, T, H, W)
            x = x[:, :, -1]  # 最後のフレーム: (B, C, H, W)

        # 元の入力サイズを保存
        _, _, h, w = x.shape

        # 512x512にアップサンプリング (CoaTは大きな入力を期待)
        x = F.interpolate(x, scale_factor=2, mode="bicubic").clip(0, 1)

        # ImageNet正規化 (CoaTが期待する入力)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # CoaTエンコーダーで特徴抽出
        # 出力: {'x1_nocls', 'x2_nocls', 'x3_nocls', 'x4_nocls'}
        encs = self.enc(x)

        # 最終ステージ(x4_nocls)の特徴量のみを使用
        x4 = encs["x4_nocls"]  # (B, 512, 16, 16) for mini/small, (B, 512, 16, 16) for medium

        # チャンネル削減
        x = self.reduce(x4)  # (B, 32, 16, 16)

        # バイリニア補間で元のサイズに戻す
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

        # 最終出力
        x = self.final(x)  # (B, 1, 256, 256)

        return x


class CoaT_Simple(nn.Module):
    """
    CoaTエンコーダーベースのシンプルなセグメンテーションモデル

    ResNet18_Simpleと同様の構造:
    - CoaT-Liteをエンコーダーとして使用
    - シンプルなデコーダー (畳み込み転置層)
    - U-Netスキップ接続なし、LSTM/FPN/UnetBlockなし

    アーキテクチャ: mini, small, medium から選択可能
    """

    def __init__(self, pre=None, arch="medium", num_classes=1, **kwargs):
        super().__init__()

        # CoaT-Liteエンコーダーの選択
        if arch == "mini":
            self.enc = coat_lite_mini(return_interm_layers=True)
            nc = [64, 128, 320, 512]
        elif arch == "small":
            self.enc = coat_lite_small(return_interm_layers=True)
            nc = [64, 128, 320, 512]
        elif arch == "medium":
            self.enc = coat_lite_medium(return_interm_layers=True)
            nc = [128, 256, 320, 512]
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        # 事前学習済み重みの読み込み（オプション）
        if pre is not None:
            sd = torch.load(pre, map_location="cpu", weights_only=False)
            if "model" in sd:
                sd = sd["model"]
            print(self.enc.load_state_dict(sd, strict=False))

        # シンプルなデコーダー (単純なアップサンプリング + 畳み込み)
        # 512x512入力に対応するため、5段の転置畳み込みでアップサンプリング
        # 16x16 -> 512x512 (32倍) -> 最後に256x256にダウンサンプリング
        self.decoder = nn.Sequential(
            # 16x16, 512 -> 32x32, 256
            nn.ConvTranspose2d(nc[-1], 256, kernel_size=4, stride=2, padding=1),
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
            # 256x256, 32 -> 512x512, 16
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # 512x512, 16 -> 512x512, num_classes
            nn.Conv2d(16, num_classes, kernel_size=1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W) or (B, C, H, W)
               - 時系列データの場合は最後のフレームのみ使用

        Returns:
            output: (B, num_classes, H, W) = (batch, 1, 256, 256)
        """
        # 時系列データの場合は最後のフレームのみ使用
        if x.dim() == 5:  # (B, C, T, H, W)
            x = x[:, :, -1]  # 最後のフレーム: (B, C, H, W)

        # 元の入力サイズを保存
        _, _, h, w = x.shape

        # 512x512にアップサンプリング (CoaTは大きな入力を期待)
        x = F.interpolate(x, scale_factor=2, mode="bicubic").clip(0, 1)

        # ImageNet正規化 (CoaTが期待する入力)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # CoaTエンコーダーで特徴抽出
        encs = self.enc(x)

        # 最終ステージ(x4_nocls)の特徴量のみを使用
        x4 = encs["x4_nocls"]  # (B, 512, 16, 16)

        # デコーダー (512x512にアップサンプリング)
        x = self.decoder(x4)  # (B, 1, 512, 512)

        # 元の入力サイズ(256x256)にダウンサンプリング
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

        return x


if __name__ == "__main__":
    # GPUが利用可能かチェック
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("=" * 60)
    print("CoaT_Simpler Test")
    print("=" * 60)

    # テスト1: CoaT_Simpler (mini)
    print("\n--- Test 1: CoaT_Simpler (arch=mini) ---")
    model = CoaT_Simple(arch="mini").to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 時系列入力
    print("\n--- Test 1-1: Temporal input (B, C, T, H, W) ---")
    x = torch.randn(2, 3, 5, 256, 256).to(device)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 1, 256, 256)

    # 単一フレーム
    print("\n--- Test 1-2: Single frame (B, C, H, W) ---")
    x = torch.randn(2, 3, 256, 256).to(device)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 1, 256, 256)

    print("\n" + "=" * 60)
    print("CoaT_SimplerFCN Test")
    print("=" * 60)

    # テスト2: CoaT_SimplerFCN (mini)
    print("\n--- Test 2: CoaT_SimplerFCN (arch=mini) ---")
    model2 = CoaT_SimplerFCN(arch="mini").to(device)

    total_params = sum(p.numel() for p in model2.parameters())
    trainable_params = sum(p.numel() for p in model2.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    x = torch.randn(2, 3, 5, 256, 256).to(device)
    y = model2(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 1, 256, 256)

    # テスト3: 他のアーキテクチャ
    print("\n" + "=" * 60)
    print("Architecture Comparison")
    print("=" * 60)

    for arch in ["mini", "small", "medium"]:
        print(f"\n--- {arch.upper()} ---")
        model_temp = CoaT_SimplerFCN(arch=arch).to(device)
        total = sum(p.numel() for p in model_temp.parameters())
        print(f"Parameters: {total:,}")

    print("\n✓ All tests passed!")
