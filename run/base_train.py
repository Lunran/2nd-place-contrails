import matplotlib.pyplot as plt
from fastai.vision.all import *


def main():
    # サンプルデータのダウンロード (CamVidデータセット)
    path = untar_data(URLs.CAMVID_TINY)
    path_im = path / "images"
    path_lbl = path / "labels"
    fnames = get_files(path_im, extensions=".png")
    # ファイル名からマスク画像を取得する関数
    get_msk = lambda o: path_lbl / f"{o.stem}_P{o.suffix}"
    # クラス名（ラベル）のリスト
    codes = np.loadtxt(path / "codes.txt", dtype=str)

    dls = SegmentationDataLoaders.from_label_func(
        path_im,
        bs=8,
        fnames=fnames,
        label_func=get_msk,
        codes=codes,
        item_tfms=[Resize(224)],  # 画像サイズ
        batch_tfms=[Normalize.from_stats(*imagenet_stats)],  # ImageNet統計量で正規化
    )

    # UNet構造でResNet18をバックボーンに使用
    learn = unet_learner(dls, resnet18, metrics=foreground_acc)

    # 最適な学習率を見つける
    # learn.lr_find()

    image = PILImage.create(path_im / fnames[0])
    mask = PILImage.create(get_msk(fnames[0]))
    pred_before, _, _ = learn.predict(image)

    # ファインチューニングの実行
    learn.fine_tune(5, base_lr=1e-3)
    # 予測結果の表示（左：元画像、中：正解、右：予測）
    learn.show_results(max_n=4, figsize=(12, 10))

    # 特定の画像で推論
    pred_after, _, _ = learn.predict(image)

    images_and_titles = [
        (image, "Original"),
        (mask, "Ground Truth"),
        (pred_before, "Prediction Before Training"),
        (pred_after, "Prediction After Training"),
    ]
    fig, axes = plt.subplots(1, len(images_and_titles), figsize=(15, 5))
    for ax, (img, title) in zip(axes, images_and_titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("data/predictions.png")
    plt.close()

    print("finished")


if __name__ == "__main__":
    main()
