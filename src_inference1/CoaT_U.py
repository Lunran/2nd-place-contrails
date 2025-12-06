import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from .coat import CoaT,coat_lite_mini,coat_lite_small,coat_lite_medium
from .layers import *

class CoaT_U(nn.Module):
    def __init__(self, pre=None, arch='medium', num_classes=1, ps=0, **kwargs):
        super().__init__()
        in_chans = 3
        if arch == 'mini': 
            self.enc = coat_lite_mini(return_interm_layers=True)
            nc = [64,128,320,512]
        elif arch == 'small': 
            self.enc = coat_lite_small(return_interm_layers=True)
            nc = [64,128,320,512]
        elif arch == 'medium': 
            self.enc = coat_lite_medium(return_interm_layers=True)
            nc = [128,256,320,512]
        else: raise Exception('Unknown model') 
        
        if pre is not None:
            sd = torch.load(pre, weights_only=False)['model']
            print(self.enc.load_state_dict(sd,strict=False))
        
        self.dec4 = UnetBlock(nc[-1],nc[-2],384)
        self.dec3 = UnetBlock(384,nc[-3],192)
        self.dec2 = UnetBlock(192,nc[-4],96)
        self.fpn = FPN([nc[-1],384,192],[32]*3)
        self.drop = nn.Dropout2d(ps)
        self.final_conv = nn.Sequential(UpBlock(96+32*3, num_classes, blur=True))
        self.up_result=1
        
        # 出力層のバイアスを負に初期化 (稀な陽性クラス対策)
        for m in self.final_conv.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, -3.0)
    
    def forward(self, x):
        # 時系列データの場合は最後のフレームを使用
        if len(x.shape) == 5: x = x[:,:,-1]
        x = F.interpolate(x,scale_factor=2,mode='bicubic').clip(0,1)
        
        # ImageNet正規化 (CoaTが期待する入力)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        
        encs = self.enc(x)
        encs = [encs[k] for k in encs]
        dec4 = encs[-1]
        dec3 = self.dec4(dec4,encs[-2])
        dec2 = self.dec3(dec3,encs[-3])
        dec1 = self.dec2(dec2,encs[-4])
        x = self.fpn([dec4, dec3, dec2], dec1)
        x = self.final_conv(self.drop(x))
        if self.up_result != 0: x = F.interpolate(x,scale_factor=self.up_result,mode='bilinear')
        return x

