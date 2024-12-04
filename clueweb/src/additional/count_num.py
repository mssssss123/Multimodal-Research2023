import argparse
import base64
import io
import os
import warnings

import clip
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import torch
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default='/data4/clip-pretrain/clue_webdata',
                        help="Path to data file")

    args = parser.parse_args()

    parquet_files = [file for file in os.listdir(args.input_path) if file.endswith('.parquet')]
    parquet_files.sort()
    file_a = parquet_files[:490]
    file_b = parquet_files[490:]

    total_num = 0
    for file in file_a:
        input_file_path = os.path.join(args.input_path, file)
        df = pd.read_parquet(input_file_path)
        num = len(df)
        total_num += num

    print(f"total num: {total_num}")
    total_num = 0
    for file in file_b:
        input_file_path = os.path.join(args.input_path, file)
        df = pd.read_parquet(input_file_path)
        num = len(df)
        total_num += num
    print(f"total num: {total_num}")
    print('finish')

if __name__ == '__main__':
    main()