import torch
import torch.nn as nn
import torch.nn.functional as F

from .coat import coat_lite_medium
from .layers import FPN, LSTM_block, UnetBlock, UpBlock


class BaseCoatULSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = coat_lite_medium(return_interm_layers=True)
        enc_dims = [128, 256, 320, 512]
        dec_dims = [96, 192, 384, 512]
        fpn_dims = [32, 32, 32]
        self.lstm = nn.ModuleList(
            [
                LSTM_block(enc_dims[-2]),
                LSTM_block(enc_dims[-1]),
            ]
        )
        self.dec4 = UnetBlock(dec_dims[-1], enc_dims[-2], dec_dims[-2])
        self.dec3 = UnetBlock(dec_dims[-2], enc_dims[-3], dec_dims[-3])
        self.dec2 = UnetBlock(dec_dims[-3], enc_dims[-4], dec_dims[-4])
        self.fpn = FPN([dec_dims[-1], dec_dims[-2], dec_dims[-3]], fpn_dims)
        self.final_conv = UpBlock(dec_dims[-4] + sum(fpn_dims), 1, blur=True)

    def forward(self, x):
        # 入力: (B, C, T, H, W) = (1, 3, 5, 256, 256) 正規化済み
        # Step 1: 時間軸展開後
        # (B*T, C, H, W) = (1*5, 3, 256, 256) = (5, 3, 256, 256)
        T = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)

        # Step 2: bicubic補間後
        # (B*T, C, H, W) = (5, 3, 256*2, 256*2) = (5, 3, 512, 512)
        x = F.interpolate(x, scale_factor=2, mode="bicubic").clip(0, 1)
        
        # ImageNet正規化 (CoaTが期待する入力)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # Step 3: CoaT各層出力
        # enc[0]: (B, T, C, H, W) = (1, 5, 128, 128, 128)  # 512/4=128, 4=patch_embed1.patch_size
        # enc[1]: (B, T, C, H, W) = (1, 5, 256, 64, 64)    # 512/8=64, 8=patch_embed2.patch_size*patch_embed1.patch_size
        # enc[2]: (B, T, C, H, W) = (1, 5, 320, 32, 32)    # 512/16=32, 16=patch_embed3.patch_size*patch_embed2.patch_size
        # enc[3]: (B, T, C, H, W) = (1, 5, 512, 16, 16)    # 512/32=16, 32=patch_embed4.patch_size*patch_embed3.patch_size
        encs = self.enc(x)  # dict
        encs = [enc.view(-1, T, *enc.shape[1:]) for enc in encs.values()]

        # Step 4: LSTM処理後
        # enc[0]: (B, C, H, W) = (1, 128, 128, 128)
        # enc[1]: (B, C, H, W) = (1, 256, 64, 64)
        # enc[2]: (B, C, H, W) = (1, 320, 32, 32)
        # enc[3]: (B, C, H, W) = (1, 512, 16, 16)
        for i in range(-2, 0):
            encs[i] = self.lstm[i](encs[i])
        encs = [enc[:, -1] for enc in encs]

        # Step 5: デコーダー各層
        # dec4: (B, C, H, W) = (1, 512, 16, 16)
        # dec3: (B, C, H, W) = (1, 384, 32, 32)
        # dec2: (B, C, H, W) = (1, 192, 64, 64)
        # dec1: (B, C, H, W) = (1, 96, 128, 128)
        dec4 = encs[-1]
        dec3 = self.dec4(dec4, encs[-2])
        dec2 = self.dec3(dec3, encs[-3])
        dec1 = self.dec2(dec2, encs[-4])

        # Step 6: FPN出力
        # (B, C, H, W) = (1, 32*3+96, 128, 128) = (1, 192, 128, 128)
        x = self.fpn([dec4, dec3, dec2], dec1)

        # Step 7: 最終出力
        # (B, C, H, W) = (1, num_classes=1, 256, 256)
        x = self.final_conv(x)

        return x


if __name__ == "__main__":
    model = BaseCoatULSTM()
    x = torch.randn(1, 3, 5, 256, 256)
    y = model(x)
    print(y.shape)
