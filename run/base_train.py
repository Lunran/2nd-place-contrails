import argparse

import matplotlib.pyplot as plt
from fastai.vision.all import *

import wandb
from src_inference1.data import ContrailsDatasetV1Single, get_aug
from src_inference1.utils import F_th

MODEL = "ResNet18_U"
EPOCHS = 1
BS = 16


class WandbCallback(Callback):
    """WandB統合用のfastai Callback（簡略版）"""

    def __init__(self):
        super().__init__()
        self._finished = False  # after_fitが複数回呼ばれるのを防ぐフラグ

    def after_batch(self):
        """バッチ終了後: 損失と学習率を記録"""

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
                }

                if hasattr(self.learn, "recorder") and len(self.learn.recorder.values) > 0:
                    last_values = self.learn.recorder.values[-1]
                    log_dict["train/loss"] = last_values[0]
                    log_dict["val/loss"] = last_values[1]
                    log_dict["val/f_th"] = last_values[2]

                wandb.log(log_dict)
                print(f"Logged to WandB: {log_dict}", flush=True)
            except Exception as e:
                print(f"WandB epoch logging error: {e}", flush=True)

    def after_fit(self):
        """学習終了後: 閾値曲線と予測画像を記録してWandBを終了"""
        # 複数回呼ばれるのを防ぐ
        if self._finished:
            return
        self._finished = True

        print("Finishing WandB run...", flush=True)
        if wandb.run is None:
            return

        try:
            # 1. 閾値曲線を記録（現在の学習結果から）
            if hasattr(self, "metrics") and len(self.metrics) > 0:
                metrics_obj = self.metrics[0]  # F_th()インスタンス
                if hasattr(metrics_obj, "ths") and hasattr(metrics_obj, "dices"):
                    # wandb.plotで閾値 vs Diceの曲線を作成
                    data = [[th, dice] for th, dice in zip(metrics_obj.ths, metrics_obj.dices.numpy())]
                    table = wandb.Table(data=data, columns=["threshold", "dice"])
                    wandb.log(
                        {
                            "threshold_curve": wandb.plot.line(
                                table, "threshold", "dice", title="Threshold vs Dice Score"
                            )
                        }
                    )

                    # 最適閾値を記録
                    optimal_idx = metrics_obj.dices.argmax()
                    optimal_th = metrics_obj.ths[optimal_idx]
                    optimal_dice = metrics_obj.dices[optimal_idx].item()
                    wandb.log(
                        {
                            "optimal_threshold": optimal_th,
                            "optimal_dice": optimal_dice,
                        }
                    )
                    print(f"Threshold curve logged to WandB (optimal: {optimal_th:.3f}, dice: {optimal_dice:.4f})")

            # 2. 予測画像サンプルを記録（12枚）
            self.learn.model.eval()
            val_dl = self.dls.valid
            images_logged = 0
            wandb_images = []

            with torch.no_grad():
                for batch_idx, (xb, yb) in enumerate(val_dl):
                    if images_logged >= 12:
                        break

                    # 予測を取得
                    preds = torch.sigmoid(self.learn.model(xb))

                    # バッチ内の各サンプルを処理
                    for i in range(xb.shape[0]):
                        if images_logged >= 12:
                            break

                        # 入力画像（中央のタイムステップを使用）
                        # xb shape: (B, C, H, W) -> (B, 3, 256, 256)
                        input_img = xb[i, :, :, :].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
                        input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)

                        # Ground truth
                        gt_mask = yb[i, 0].cpu().numpy()  # (H, W)

                        # 予測マスク
                        pred_mask = preds[i, 0].cpu().numpy()  # (H, W)

                        # matplotlibで3列レイアウトの画像を作成
                        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                        # 左: 入力画像
                        axes[0].imshow(input_img)
                        axes[0].set_title("Input")
                        axes[0].axis("off")

                        # 中央: Ground Truth
                        axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
                        axes[1].set_title("Ground Truth")
                        axes[1].axis("off")

                        # 右: 予測Probability
                        im = axes[2].imshow(pred_mask, cmap="viridis", vmin=0, vmax=1)
                        axes[2].set_title("Prediction Probability")
                        axes[2].axis("off")
                        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

                        plt.tight_layout()

                        # 図をnumpy配列に変換
                        fig.canvas.draw()
                        buf = np.array(fig.canvas.buffer_rgba())
                        img_array = buf[:, :, :3]  # RGBA -> RGB

                        # WandB Imageを作成
                        wandb_images.append(
                            wandb.Image(
                                img_array,
                                caption=f"Sample {images_logged + 1}",
                            )
                        )

                        # メモリ解放
                        plt.close(fig)
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


def main(args):
    print("\n[INFO] Initializing WandB manually...", flush=True)
    wandb_config_path = Path.home() / ".wandb" / "wandb.json"
    with open(wandb_config_path, "r") as f:
        wandb_config = json.load(f)
        os.environ["WANDB_API_KEY"] = wandb_config.get("api_key", "")
    wandb_run = wandb.init(
        project="2nd-place-contrails",
        mode="online",
        config={
            "model": MODEL,
            "size": args.size,
            "epochs": EPOCHS,
            "batch_size": BS,
            # "gradient_accumulation": int(16 / BS + 0.5),
            "learning_rate": 1e-3,  # 3.5e-4,
            # "pct_start": 0.1,
            # "loss_function": "BCE + 0.01 * Lovasz",
            # "optimizer": "WrapperOver9000 (RAdam + LAMB + Lookahead)",
            # "scheduler": "OneCycle",
            # "gradient_clip": 3.0,
            # "seed": 2023,
        },
    )
    print(f"✓ WandB initialized: {wandb_run.name}", flush=True)
    print(f"✓ View run at: {wandb_run.url}\n", flush=True)

    ds_train = ContrailsDatasetV1Single("data", train=True, tfms=None, size=args.size)
    ds_val = ContrailsDatasetV1Single("data", train=False, tfms=None, size=args.size)
    dls = ImageDataLoaders.from_dsets(ds_train, ds_val, bs=BS, num_workers=4, pin_memory=False).cuda()

    learn = unet_learner(
        dls,
        resnet18,
        n_out=1,
        loss_func=F.binary_cross_entropy_with_logits,
        metrics=F_th(),
        cbs=[
            WandbCallback(),
        ],
    )
    learn.fine_tune(EPOCHS, base_lr=1e-3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=None, help="Dataset size for debugging")
    args = parser.parse_args()
    main(args)
