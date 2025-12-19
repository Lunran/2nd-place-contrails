"""
データ前処理ユーティリティ

このモジュールは、OpenContrailsデータセットの処理と変換を行います。
主な機能:
- 複数のバンドデータ（band_08-16）の読み込み
- マスクデータ（human_pixel_masks, human_individual_masks）の読み込み
- 疑似カラー画像の生成（Ashカラースキーム）
- HDF5形式への変換・集約
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm

# 定数定義
N_TIMES_BEFORE = 4
N_TIMES_AFTER = 3
N_TIMES_TOTAL = N_TIMES_BEFORE + N_TIMES_AFTER + 1  # 8

# 疑似カラー画像の境界値（Ashカラースキーム改良版）
_T11_BOUNDS = (243, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)

# 利用可能なバンド
BAND_NUMBERS = [8, 9, 10, 11, 12, 13, 14, 15, 16]


def load_band_data(base_dir: str, record_id: str, band_num: int) -> np.ndarray:
    """
    指定したバンドのデータを読み込む

    Args:
        base_dir: データディレクトリのパス
        record_id: レコードID
        band_num: バンド番号（8-16）

    Returns:
        shape (H, W, T) のnumpy配列
    """
    file_path = os.path.join(base_dir, record_id, f"band_{band_num:02d}.npy")
    with open(file_path, "rb") as f:
        data = np.load(f)
    return data


def load_mask_data(base_dir: str, record_id: str, mask_type: str = "pixel") -> Optional[np.ndarray]:
    """
    マスクデータを読み込む

    Args:
        base_dir: データディレクトリのパス
        record_id: レコードID
        mask_type: 'pixel' または 'individual'

    Returns:
        マスクの numpy配列。ファイルが存在しない場合はNone
    """
    if mask_type == "pixel":
        file_name = "human_pixel_masks.npy"
    elif mask_type == "individual":
        file_name = "human_individual_masks.npy"
    else:
        raise ValueError(f"Invalid mask_type: {mask_type}")

    file_path = os.path.join(base_dir, record_id, file_name)
    if not os.path.exists(file_path):
        return None

    with open(file_path, "rb") as f:
        data = np.load(f)
    return data


def normalize_range(data: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
    """
    データを [0, 1] の範囲に正規化する

    Args:
        data: 入力データ
        bounds: (min_value, max_value) のタプル

    Returns:
        正規化されたデータ
    """
    return (data - bounds[0]) / (bounds[1] - bounds[0])


def get_record_ids(base_dir: str) -> List[str]:
    """
    データディレクトリ内のすべてのレコードIDを取得

    Args:
        base_dir: データディレクトリのパス

    Returns:
        レコードIDのリスト
    """
    base_path = Path(base_dir)
    record_ids = [d.name for d in base_path.iterdir() if d.is_dir()]
    return sorted(record_ids)


def get_false_color(band11: np.ndarray, band14: np.ndarray, band15: np.ndarray, as_uint8: bool = True) -> np.ndarray:
    """
    偽色データを生成（data.pyのget_false_colorと同じロジック）

    R: band_15 - band_14 (TDIFF)
    G: band_14 - band_11 (CLOUD_TOP_TDIFF)
    B: band_14 (T11)

    Args:
        band11: Band 11データ (H, W, T)
        band14: Band 14データ (H, W, T)
        band15: Band 15データ (H, W, T)
        as_uint8: True の場合、uint8 (0-255) で返す。False の場合、float32 (0-1) で返す

    Returns:
        shape (H, W, T, 3) の偽色データ
    """
    # 偽色チャネルを計算
    r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
    g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(band14, _T11_BOUNDS)

    # スタックしてクリップ
    false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)

    if as_uint8:
        return (false_color * 255).clip(0, 255).astype(np.uint8)
    else:
        return false_color.astype(np.float32)


def process_single_record(base_dir: str, record_id: str, use_uint8: bool = True) -> Dict[str, np.ndarray]:
    """
    1つのレコードを処理してデータを返す

    Args:
        base_dir: データディレクトリのパス
        record_id: レコードID
        use_uint8: uint8に量子化するかどうか

    Returns:
        処理されたデータの辞書（band_11, band_14, band_15, pixel_maskのみ）
    """
    result = {}

    # band 11, 14, 15のみを読み込み
    band11 = load_band_data(base_dir, record_id, 11)
    band14 = load_band_data(base_dir, record_id, 14)
    band15 = load_band_data(base_dir, record_id, 15)

    # 偽色データを生成 (H, W, T, 3)
    # R: band_15 - band_14, G: band_14 - band_11, B: band_14
    result["bands"] = get_false_color(band11, band14, band15, as_uint8=use_uint8)

    # マスクデータを読み込み（pixel_maskのみ）
    pixel_mask = load_mask_data(base_dir, record_id, "pixel")
    if pixel_mask is not None:
        if use_uint8:
            result["pixel_mask"] = (pixel_mask * 255).clip(0, 255).astype(np.uint8)
        else:
            result["pixel_mask"] = pixel_mask.astype(np.float32)

    return result


def convert_to_hdf5(
    base_dir: str, output_path: str, compression: str = "gzip", compression_opts: int = 4, use_uint8: bool = True
) -> None:
    """
    データディレクトリ全体をHDF5形式に変換

    Args:
        base_dir: データディレクトリのパス (train, validation, test)
        output_path: 出力するHDF5ファイルのパス
        compression: HDF5の圧縮方式
        compression_opts: 圧縮レベル（1-9、高いほど圧縮率が高い）
        use_uint8: データをuint8に量子化するかどうか
    """
    record_ids = get_record_ids(base_dir)

    if len(record_ids) == 0:
        raise ValueError(f"No records found in {base_dir}")

    print(f"Processing {len(record_ids)} records from {base_dir}")
    print(f"Output: {output_path}")
    print(f"Data type: {'uint8' if use_uint8 else 'float32'}")

    with h5py.File(output_path, "w") as hf:
        # 各レコードを処理
        for record_id in tqdm(record_ids, desc="Converting to HDF5"):
            try:
                data = process_single_record(base_dir, record_id, use_uint8=use_uint8)

                # HDF5のグループを作成
                grp = hf.create_group(record_id)

                # データセットを作成
                for key, value in data.items():
                    grp.create_dataset(key, data=value, compression=compression, compression_opts=compression_opts)

            except Exception as e:
                print(f"\nError processing {record_id}: {e}")
                continue

    print(f"\nSuccessfully created {output_path}")

    # ファイルサイズを表示
    file_size = os.path.getsize(output_path) / (1024**3)  # GB
    print(f"File size: {file_size:.2f} GB")


def load_from_hdf5(hdf5_path: str, record_id: str) -> Dict[str, np.ndarray]:
    """
    HDF5ファイルから特定のレコードを読み込む

    Args:
        hdf5_path: HDF5ファイルのパス
        record_id: レコードID

    Returns:
        データの辞書
    """
    with h5py.File(hdf5_path, "r") as hf:
        if record_id not in hf:
            raise KeyError(f"Record {record_id} not found in {hdf5_path}")

        grp = hf[record_id]
        data = {key: grp[key][()] for key in grp.keys()}

    return data


def get_hdf5_record_ids(hdf5_path: str) -> List[str]:
    """
    HDF5ファイル内のすべてのレコードIDを取得

    Args:
        hdf5_path: HDF5ファイルのパス

    Returns:
        レコードIDのリスト
    """
    with h5py.File(hdf5_path, "r") as hf:
        record_ids = list(hf.keys())

    return sorted(record_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert OpenContrails dataset to HDF5 format")
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Input data directory (train, validation, or test)"
    )
    parser.add_argument("--output-file", type=str, required=True, help="Output HDF5 file path")
    parser.add_argument(
        "--compression", type=str, default="gzip", choices=["gzip", "lzf", None], help="HDF5 compression method"
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        choices=range(1, 10),
        help="Compression level (1-9, higher = better compression)",
    )
    parser.add_argument(
        "--uint8", action="store_true", default=True, help="Quantize data to uint8 for better compression"
    )

    args = parser.parse_args()

    # データディレクトリの存在確認
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    # 出力ディレクトリの作成
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 変換実行
    convert_to_hdf5(
        base_dir=args.input_dir,
        output_path=args.output_file,
        compression=args.compression,
        compression_opts=args.compression_level,
        use_uint8=args.uint8,
    )
