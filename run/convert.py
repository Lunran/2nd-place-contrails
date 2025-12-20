#!/usr/bin/env python3
"""
データを HDF5 形式に変換するスクリプト

使用方法:
    python run/convert.py --data-type validation
    python run/convert.py --data-type train
    python run/convert.py --data-type train --force  # 上書き確認をスキップ
"""

import argparse
import os
import sys

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocess import convert_to_hdf5


def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="データをHDF5形式に変換")
    parser.add_argument(
        "--data-type",
        type=str,
        required=True,
        choices=["train", "validation"],
        help="変換するデータタイプ (train または validation)",
    )

    args = parser.parse_args()

    # パス設定（動的に構築）
    input_dir = os.path.join(project_root, "data", args.data_type)
    output_file = os.path.join(project_root, "data", f"{args.data_type}.hdf5")

    # 既存ファイルの確認
    if os.path.exists(output_file):
        print(f"出力ファイルが既に存在します: {output_file}")
        return

    print("=" * 80)
    print(f"{args.data_type.capitalize()} データを HDF5 形式に変換")
    print("=" * 80)
    print()

    # 変換実行
    convert_to_hdf5(base_dir=input_dir, output_path=output_file, compression="gzip", compression_opts=4)

    print()
    print("=" * 80)
    print("変換完了")
    print("=" * 80)


if __name__ == "__main__":
    main()
