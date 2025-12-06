#!/usr/bin/env python3
"""
CoaT-Lite Medium ImageNet事前学習済み重みをダウンロードするスクリプト
experimentsディレクトリに保存
"""
import sys
import urllib.request
from pathlib import Path


def download_coat_weights(output_dir: str = "experiments"):
    """
    CoaT-Lite MediumのImageNet事前学習済み重みをダウンロード

    Args:
        output_dir: 保存先ディレクトリ
    """
    # 出力ディレクトリを作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ダウンロードする重みファイルのリスト
    weights = {
        "coat_lite_medium_a750cd63.pth": {
            "url": "https://vcl.ucsd.edu/coat/pretrained/coat_lite_medium_a750cd63.pth",
            "description": "CoaT-Lite Medium (224x224, Acc@1=83.6%)",
        },
        "coat_lite_medium_384x384_f9129688.pth": {
            "url": "https://vcl.ucsd.edu/coat/pretrained/coat_lite_medium_384x384_f9129688.pth",
            "description": "CoaT-Lite Medium (384x384, Acc@1=84.5%) [推奨]",
        },
    }

    downloaded_files = []

    for filename, info in weights.items():
        weights_file = output_path / filename

        if weights_file.exists():
            print(f"✓ 重みファイルは既に存在します: {weights_file}")
            file_size_mb = weights_file.stat().st_size / 1024 / 1024
            print(f"  ファイルサイズ: {file_size_mb:.2f} MB")
            downloaded_files.append(str(weights_file))
            continue

        print(f"\n{info['description']}")
        print(f"ダウンロード中: {filename}...")
        print(f"URL: {info['url']}")

        try:
            # ダウンロード（進捗表示付き）
            urllib.request.urlretrieve(info["url"], weights_file)

            file_size_mb = weights_file.stat().st_size / 1024 / 1024
            print(f"✓ 重みを保存しました: {weights_file}")
            print(f"  ファイルサイズ: {file_size_mb:.2f} MB")
            downloaded_files.append(str(weights_file))

        except Exception as e:
            print(f"✗ ダウンロードに失敗しました: {e}")
            if weights_file.exists():
                weights_file.unlink()  # 不完全なファイルを削除

    return downloaded_files


if __name__ == "__main__":
    # コマンドライン引数から保存先を取得 (デフォルトはexperiments)
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "experiments"

    print("=" * 70)
    print("CoaT-Lite Medium 事前学習済み重みダウンロードスクリプト")
    print("=" * 70)

    weights_files = download_coat_weights(output_dir)

    print("\n" + "=" * 70)
    print("ダウンロード完了")
    print("=" * 70)
    print("\n使用方法:")
    print("  # 384x384版（推奨 - 512x512入力に適合）")
    print(f"  model = CoaT_SimplerFCN(")
    print(f"      pre='experiments/coat_lite_medium_384x384_f9129688.pth',")
    print(f"      arch='medium'")
    print(f"  )")
    print("\n  # 224x224版（標準）")
    print(f"  model = CoaT_SimplerFCN(")
    print(f"      pre='experiments/coat_lite_medium_a750cd63.pth',")
    print(f"      arch='medium'")
    print(f"  )")
