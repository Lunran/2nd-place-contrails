import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from src_inference1.layers import FPN, LSTM_block, UnetBlock, UpBlock

N_TIMES_BEFORE = 4


class BaseResNetU(nn.Module):

    def __init__(self, weights_path):
        super().__init__()
        backbone = resnet18(weights=None)
        state_dict = torch.load(weights_path, map_location="cpu")
        backbone.load_state_dict(state_dict)

        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4
        self.nc = [64, 64, 128, 256, 512]

        self.dec4 = UnetBlock(self.nc[-1], self.nc[-2], 384)
        self.dec3 = UnetBlock(384, self.nc[-3], 192)
        self.dec2 = UnetBlock(192, self.nc[-4], 96)
        self.fpn = FPN([self.nc[-1], 384, 192], [32] * 3)

        self.final_conv = UpBlock(96 + 32 * 3, 1, blur=True)

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W) = (batch, 3, 5, 256, 256)
               または (B, C, H, W) = (batch, 3, 256, 256)

        Returns:
            output: (B, num_classes, H, W) = (batch, 1, 256, 256)
        """
        if x.dim() == 5:
            # 時系列入力 (B, C, T, H, W) の場合、最後のフレームを抽出
            x = x[:, :, -1, :, :]  # (B, 3, 256, 256)

        i1 = F.interpolate(x, scale_factor=2, mode="bicubic").clip(0, 1)  # (B, 3, 512, 512)
        e0 = self.enc0(i1)  # (B, 64, 128, 128)
        e1 = self.enc1(e0)  # (B, 64, 128, 128)
        e2 = self.enc2(e1)  # (B, 128, 64, 64)
        e3 = self.enc3(e2)  # (B, 256, 32, 32)
        e4 = self.enc4(e3)  # (B, 512, 16, 16)
        d4 = e4
        d3 = self.dec4(d4, e3)  # (B, 384, 32, 32)
        d2 = self.dec3(d3, e2)  # (B, 192, 64, 64)
        d1 = self.dec2(d2, e1)  # (B, 96, 128, 128)
        f1 = self.fpn([d4, d3, d2], d1)  # (B, 96+32*3, 128, 128)
        o = self.final_conv(f1)  # (B, 1, 256, 256)
        return o


class BaseResNetULSTM(nn.Module):

    def __init__(self, weights_path):
        super().__init__()
        backbone = resnet18(weights=None)
        state_dict = torch.load(weights_path, map_location="cpu")
        backbone.load_state_dict(state_dict)

        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4
        self.nc = [64, 64, 128, 256, 512]

        self.lstm3 = LSTM_block(self.nc[-2])
        self.lstm4 = LSTM_block(self.nc[-1])
        self.dec4 = UnetBlock(self.nc[-1], self.nc[-2], 384)
        self.dec3 = UnetBlock(384, self.nc[-3], 192)
        self.dec2 = UnetBlock(192, self.nc[-4], 96)
        self.fpn = FPN([self.nc[-1], 384, 192], [32] * 3)

        self.final_conv = UpBlock(96 + 32 * 3, 1, blur=True)

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W) = (BS, 3, N_TIMES_BEFORE + 1, 256, 256)
            dataloaderで T = N_TIMES_BEFORE + 1 = 5 に設定していることを想定

        Returns:
            output: (B, num_classes, H, W) = (BS, 1, 256, 256)
        """
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)  # (BS*5, 3, 256, 256)

        # 5Dに対応していない操作
        i1 = F.interpolate(x, scale_factor=2, mode="bicubic").clip(0, 1)  # (BS*5, 3, 512, 512)
        e0 = self.enc0(i1)  # (BS*5, 64, 128, 128)
        e1 = self.enc1(e0)  # (BS*5, 64, 128, 128)
        e2 = self.enc2(e1)  # (BS*5, 128, 64, 64)
        e3 = self.enc3(e2)  # (BS*5, 256, 32, 32)
        e4 = self.enc4(e3)  # (BS*5, 512, 16, 16)

        e1 = e1.view(-1, N_TIMES_BEFORE + 1, e1.shape[1], e1.shape[2], e1.shape[3])  # (BS, 5, 64, 128, 128)
        e2 = e2.view(-1, N_TIMES_BEFORE + 1, e2.shape[1], e2.shape[2], e2.shape[3])  # (BS, 5, 128, 64, 64)
        e3 = e3.view(-1, N_TIMES_BEFORE + 1, e3.shape[1], e3.shape[2], e3.shape[3])  # (BS, 5, 256, 32, 32)
        e4 = e4.view(-1, N_TIMES_BEFORE + 1, e4.shape[1], e4.shape[2], e4.shape[3])  # (BS, 5, 512, 16, 16)
        l1 = e1[:, -1]  # (BS, 64, 128, 128)
        l2 = e2[:, -1]  # (BS, 128, 64, 64)
        l3 = self.lstm3(e3)[:, -1]  # (BS, 256, 32, 32)
        l4 = self.lstm4(e4)[:, -1]  # (BS, 512, 16, 16)
        d4 = l4
        d3 = self.dec4(d4, l3)  # (BS, 384, 32, 32)
        d2 = self.dec3(d3, l2)  # (BS, 192, 64, 64)
        d1 = self.dec2(d2, l1)  # (BS, 96, 128, 128)
        f1 = self.fpn([d4, d3, d2], d1)  # (BS, 96+32*3, 128, 128)
        o = self.final_conv(f1)  # (BS, 1, 256, 256)
        return o


if __name__ == "__main__":
    BS = 8
    model = BaseResNetULSTM(weight_path="data/resnet18-imagenet.pth")

    # チャンネル数を表示
    print(f"Channel dimensions: {model.nc}")

    # パラメータ数を計算
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ダミー入力でテスト (時系列)
    print("\n=== Test 1: Temporal input (B, C, T, H, W) ===")
    x = torch.randn(BS, 3, N_TIMES_TOTAL, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (BS, 1, 256, 256), f"Expected ({BS}, 1, 256, 256), got {y.shape}"
    print("\n✓ All model tests passed!")
