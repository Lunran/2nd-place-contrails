import argparse
import gc
import json
import os
from pathlib import Path

import pandas as pd

import wandb
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
EPOCHS = 12
BS = 4


class WandbCallback(Callback):
    """WandB統合用のfastai Callback（簡略版）"""

    def after_batch(self):
        """バッチ終了後: 損失と学習率を記録"""
        print("Batch finished", flush=True)

        if self.training and wandb.run is not None:
            try:
                current_lr = self.opt.hypers[0]["lr"] if hasattr(self.opt, "hypers") else None
                log_dict = {
                    "train/batch_loss": self.smooth_loss,
                    "train/epoch": self.epoch,
                    "train/batch": self.iter,
                }
                if current_lr is not None:
                    log_dict["train/learning_rate"] = current_lr
                wandb.log(log_dict)
            except Exception as e:
                pass  # サイレント

    def after_epoch(self):
        """エポック終了後: メトリクスを記録"""
        print("Epoch finished", flush=True)

        if wandb.run is not None:
            try:
                log_dict = {
                    "epoch": self.epoch,
                    "train/loss": self.recorder.log[self.epoch][0],
                    "valid/loss": self.recorder.log[self.epoch][1],
                }

                # メトリクスを記録（F_thなど）
                if len(self.recorder.log[self.epoch]) > 2:
                    log_dict["valid/f_th"] = self.recorder.log[self.epoch][2]

                wandb.log(log_dict)
                print(f"Epoch {self.epoch} logged to WandB")
            except Exception as e:
                print(f"WandB epoch logging error: {e}")

    def after_fit(self):
        """学習終了後: 閾値曲線と予測画像を記録してWandBを終了"""
        print("Finishing WandB run...", flush=True)
        if wandb.run is None:
            return

        try:
            # 1. 閾値曲線を記録
            import pandas as pd

            threshold_csv = "experiments/threshold_curve.csv"
            if os.path.exists(threshold_csv):
                df = pd.read_csv(threshold_csv)
                # wandb.plotで閾値 vs Diceの曲線を作成
                data = [[th, dice] for th, dice in zip(df["threshold"], df["dice"])]
                table = wandb.Table(data=data, columns=["threshold", "dice"])
                wandb.log(
                    {"threshold_curve": wandb.plot.line(table, "threshold", "dice", title="Threshold vs Dice Score")}
                )

                # 最適閾値を記録
                optimal_idx = df["dice"].argmax()
                optimal_th = df["threshold"].iloc[optimal_idx]
                optimal_dice = df["dice"].iloc[optimal_idx]
                wandb.log(
                    {
                        "optimal_threshold": optimal_th,
                        "optimal_dice": optimal_dice,
                    }
                )
                print(f"Threshold curve logged to WandB (optimal: {optimal_th:.3f}, dice: {optimal_dice:.4f})")

            # 2. 予測画像サンプルを記録（4枚）
            self.learn.model.eval()
            val_dl = self.dls.valid
            images_logged = 0
            wandb_images = []

            with torch.no_grad():
                for batch_idx, (xb, yb) in enumerate(val_dl):
                    if images_logged >= 4:
                        break

                    # 予測を取得
                    preds = torch.sigmoid(self.learn.model(xb))

                    # バッチ内の各サンプルを処理
                    for i in range(xb.shape[0]):
                        if images_logged >= 4:
                            break

                        # 入力画像（中央のタイムステップを使用）
                        # xb shape: (B, C, T, H, W) -> (B, 3, 5, 256, 256)
                        input_img = xb[i, :, 2, :, :].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
                        input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)

                        # Ground truth
                        gt_mask = yb[i, 0].cpu().numpy()  # (H, W)

                        # 予測マスク
                        pred_mask = preds[i, 0].cpu().numpy()  # (H, W)

                        # WandB Imageを作成（マスクをオーバーレイ）
                        wandb_images.append(
                            wandb.Image(
                                input_img,
                                masks={
                                    "predictions": {
                                        "mask_data": pred_mask,
                                        "class_labels": {0: "background", 1: "contrail"},
                                    },
                                    "ground_truth": {
                                        "mask_data": gt_mask,
                                        "class_labels": {0: "background", 1: "contrail"},
                                    },
                                },
                                caption=f"Sample {images_logged + 1}",
                            )
                        )
                        images_logged += 1

            if wandb_images:
                wandb.log({"predictions": wandb_images})
                print(f"{len(wandb_images)} prediction samples logged to WandB")

        except Exception as e:
            print(f"WandB final logging error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # WandBを終了
            print("\n[INFO] Finishing WandB run...", flush=True)
            wandb.finish()
            print("✓ WandB run finished\n", flush=True)


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
    # model = CoaT_U(pre="data/coat_lite_medium_384x384_f9129688.pth").cuda()

    # オプション2: BaseCoatULSTM（LSTM + 事前学習済みCoaT）
    # model = BaseCoatULSTM().cuda()
    # 事前学習済み重みを読み込む場合：
    # model.enc.load_state_dict(torch.load("data/coat_lite_medium_384x384_f9129688.pth")["model"], strict=False)

    # オプション3: CoaT_ULSTM（LSTM + 事前学習済みCoaT）
    model = CoaT_ULSTM(pre="data/coat_lite_medium_384x384_f9129688.pth").cuda()

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
            WandbCallback(),
        ],
        opt_func=partial(WrapperOver9000, eps=1e-4),
    )

    print("\n[INFO] Initializing WandB manually...", flush=True)
    wandb_config_path = Path.home() / ".wandb" / "wandb.json"
    with open(wandb_config_path, "r") as f:
        wandb_config = json.load(f)
        os.environ["WANDB_API_KEY"] = wandb_config.get("api_key", "")
    wandb_run = wandb.init(
        project="2nd-place-contrails",
        mode="online",
        config={
            "epochs": EPOCHS,
            "batch_size": BS,
            "gradient_accumulation": int(16 / BS + 0.5),
            "learning_rate": 3.5e-4,
            "pct_start": 0.1,
            "model": "CoaT_ULSTM",
            "loss_function": "BCE + 0.01 * Lovasz",
            "optimizer": "WrapperOver9000 (RAdam + LAMB + Lookahead)",
            "scheduler": "OneCycle",
            "gradient_clip": 3.0,
            "seed": 2023,
            "size": args.size,
        },
    )
    print(f"✓ WandB initialized: {wandb_run.name}", flush=True)
    print(f"✓ View run at: {wandb_run.url}\n", flush=True)

    # 学習率を最適化（CoaT向けに調整: config.jsonと同じ3.5e-4）
    learn.fit_one_cycle(EPOCHS, lr_max=3.5e-4, pct_start=0.1)
    torch.save(learn.model.state_dict(), MODEL_PATH)

    # 最適閾値の取得と表示
    optimal_th = metrics.optimal_threshold
    optimal_dice = metrics.value
    print(f"\n最適閾値: {optimal_th:.3f}, 最適Dice: {optimal_dice:.4f}")
    threshold_df = pd.DataFrame({"threshold": metrics.ths, "dice": metrics.dices.numpy()})
    threshold_df.to_csv("experiments/threshold_curve.csv", index=False)

    if wandb.run is not None:
        wandb.run.summary["optimal_threshold"] = optimal_th
        wandb.run.summary["optimal_dice"] = optimal_dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=None, help="Dataset size for debugging")
    args = parser.parse_args()
    main(args)
