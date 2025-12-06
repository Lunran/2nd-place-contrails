import argparse
import gc
import json
import os

from src_inference1.BaseCoatULSTM import BaseCoatULSTM
from src_inference1.CoaT_Simple import CoaT_Simple, CoaT_SimplerFCN
from src_inference1.CoaT_U import CoaT_U
from src_inference1.CoaT_ULSTM import CoaT_ULSTM
from src_inference1.data import ContrailsDatasetV0, get_aug
from src_inference1.fastai_fix import *
from src_inference1.lovasz import lovasz_hinge
from src_inference1.ResNet18_Simple import ResNet18_Simple
from src_inference1.ResNet18_U import ResNet18_U
from src_inference1.ResNet18_ULSTM import ResNet18_ULSTM
from src_inference1.utils import F_th, WrapperOver9000, seed_everything

# OOM対策: PyTorchメモリ管理の最適化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

seed_everything(2023)

# GPUメモリをクリア
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

MODEL_PATH = "experiments/CoaT_ULSTM.pth"
EPOCHS = 4
BS = 4


def loss_comb(x, y):
    return F.binary_cross_entropy_with_logits(x, y) + 0.01 * 0.5 * (
        lovasz_hinge(x, y, per_image=False) + lovasz_hinge(-x, 1 - y, per_image=False)
    )


def main(args):
    ds_train = ContrailsDatasetV0("data/", train=True, tfms=get_aug(), size=args.size)
    ds_val = ContrailsDatasetV0("data", train=False, tfms=None, size=args.size)

    data = ImageDataLoaders.from_dsets(ds_train, ds_val, bs=BS, num_workers=4, pin_memory=True).cuda()

    # モデル選択（ImageNet正規化対応版）
    # オプション1: CoaT_U（U-Net + 事前学習済みCoaT）
    # model = CoaT_U(pre="experiments/coat_lite_medium_384x384_f9129688.pth").cuda()
    
    # オプション2: BaseCoatULSTM（LSTM + 事前学習済みCoaT）
    # model = BaseCoatULSTM().cuda()
    # 事前学習済み重みを読み込む場合：
    # model.enc.load_state_dict(torch.load("experiments/coat_lite_medium_384x384_f9129688.pth")["model"], strict=False)
    
    # オプション3: CoaT_ULSTM（LSTM + 事前学習済みCoaT）
    model = CoaT_ULSTM(pre="experiments/coat_lite_medium_384x384_f9129688.pth").cuda()

    metrics = F_th()

    learn = Learner(
        data,
        model,
        path="experiments",
        loss_func=loss_comb,
        metrics=metrics,
        cbs=[
            GradientClip(3.0),  # CoaT向けに最適化
            GradientAccumulation(int(16 / BS + 0.5)),
            CSVLogger(),
            SaveModelCallback(monitor="f_th"),
        ],
        opt_func=partial(WrapperOver9000, eps=1e-4),
    )

    # 学習率を最適化（CoaT向けに調整: config.jsonと同じ3.5e-4）
    learn.fit_one_cycle(EPOCHS, lr_max=3.5e-4, pct_start=0.1)
    torch.save(learn.model.state_dict(), MODEL_PATH)

    # 最適閾値の取得と表示
    optimal_th = metrics.optimal_threshold
    optimal_dice = metrics.value
    print(f"\n最適閾値: {optimal_th:.3f}, 最適Dice: {optimal_dice:.4f}")

    # 閾値探索結果をCSVに保存
    import pandas as pd

    threshold_df = pd.DataFrame({"threshold": metrics.ths, "dice": metrics.dices.numpy()})
    threshold_df.to_csv("experiments/threshold_curve.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=None, help="Dataset size for debugging")
    args = parser.parse_args()
    main(args)
