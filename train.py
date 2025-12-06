import argparse
import json

from src_inference1.CoaT_ULSTM import CoaT_ULSTM
from src_inference1.CoaT_UT import CoaT_UT
from src_inference1.data import ContrailsDatasetV0, get_aug
from src_inference1.fastai_fix import *
from src_inference1.lovasz import lovasz_hinge
from src_inference1.NeXtViT_ULSTM import NeXtViT_ULSTM
from src_inference1.SAM import SAM_U, SAM_USA, SAM_UV1, SAM_UV2, SAM_UV3
from src_inference1.utils import F_th, WrapperOver9000, seed_everything


def read_config_file(file_path):
    with open(file_path, "r") as f:
        config_data = json.load(f)
    return config_data


def loss_comb(x, y):
    return F.binary_cross_entropy_with_logits(x, y) + 0.01 * 0.5 * (
        lovasz_hinge(x, y, per_image=False) + lovasz_hinge(-x, 1 - y, per_image=False)
    )


class BatchCSVLogger(CSVLogger):
    def __init__(self, fname="history.csv", append=False, log_every=10):
        super().__init__(fname, append)
        self.log_every = log_every
        self.batch_log_file = f"experiments/batch_{self.fname}"

    def after_batch(self):
        if self.training and self.iter % self.log_every == 0:
            try:
                current_lr = self.opt.hypers[0]["lr"] if hasattr(self.opt, "hypers") else "N/A"
                log_line = f"{self.epoch},{self.iter},{self.smooth_loss:.6f},{current_lr}\n"

                with open(self.batch_log_file, "a") as f:
                    f.write(log_line)

                # コンソールにも出力
                print(f"Epoch {self.epoch}, Batch {self.iter}: Loss={self.smooth_loss:.6f}, LR={current_lr}")

            except Exception as e:
                print(f"ログ記録エラー: {e}")


def train(config):
    ds_train = ContrailsDatasetV0(config["PATH"], train=True, tfms=get_aug(), size=1000)
    ds_val = ContrailsDatasetV0(config["PATH"], train=False, tfms=None, size=1000)
    data = ImageDataLoaders.from_dsets(
        ds_train,
        ds_val,
        bs=config["BS"],
        num_workers=config["NUM_WORKERS"],
        pin_memory=True,
    ).cuda()

    model = config["MODEL"]
    if config["WEIGHTS"]:
        print("Loading weights from ...", config["WEIGHTS"])
        model.load_state_dict(torch.load(config["WEIGHTS"]))
    model = nn.DataParallel(model)
    model = model.cuda()
    learn = Learner(
        data,
        model,
        path=config["OUT"],
        loss_func=config["LOSS_FUNC"],
        metrics=[config["METRIC"]],
        cbs=[
            GradientClip(3.0),
            GradientAccumulation(int(16 / config["BS"] + 0.5)),
            BatchCSVLogger(),
            SaveModelCallback(monitor="f_th"),
        ],
        opt_func=partial(WrapperOver9000, eps=1e-4),
    ).to_fp16()

    learn.fit_one_cycle(config["EPOCHS"], lr_max=config["LR_MAX"], pct_start=config["PCT_START"])
    torch.save(
        learn.model.module.state_dict(),
        os.path.join(config["OUT"], f'{config["FNAME"]}_{config["FOLD"]}.pth'),
    )


def main():
    parser = argparse.ArgumentParser(description="Create a model from a JSON config file.")
    parser.add_argument("config_file", type=str, help="Path to the JSON config file.")
    parser.add_argument(
        "configs",
        nargs="*",
        metavar=("KEY", "VALUE"),
        help="The JSON config key to override and its new value.",
    )

    args = parser.parse_args()
    config_file_path = args.config_file

    config_data = read_config_file(config_file_path)

    if args.configs:
        for config_key, config_value in zip(args.configs[::2], args.configs[1::2]):
            keys = config_key.split(".")
            last_key = keys.pop()

            current_data = config_data
            for key in keys:
                current_data = current_data[key]

            try:
                value = json.loads(config_value)
            except json.JSONDecodeError:
                value = config_value

            current_data[last_key] = value

    print("Training with the following configuration:")
    print(json.dumps(config_data, indent=4))
    print("_______________________________________________________")

    config_data["MODEL"] = getattr(sys.modules[__name__], config_data["MODEL"])()
    config_data["LOSS_FUNC"] = getattr(sys.modules[__name__], config_data["LOSS_FUNC"])
    config_data["METRIC"] = getattr(sys.modules[__name__], config_data["METRIC"])()

    seed_everything(config_data["SEED"])
    os.makedirs(config_data["OUT"], exist_ok=True)
    train(config_data)


if __name__ == "__main__":
    main()
