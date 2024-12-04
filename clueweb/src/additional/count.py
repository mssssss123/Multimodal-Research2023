import argparse
import io
import json
import os

import pandas as pd
from PIL import Image
import pyarrow.parquet as pq



def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--data_files", type=str, default='/data3/clip-pretrain/clue_webdata', help="Path to data file")
    args = parser.parse_args()

    total_samples = 0
    for filename in os.listdir(args.data_files):
        if filename.endswith('.parquet'):
            file_path = os.path.join(args.data_files, filename)

            # 使用pyarrow来读取Parquet文件
            table = pq.read_table(file_path)

            # 获取当前Parquet文件中的样本数
            num_samples = len(table)

            # 累加到总样本数
            total_samples += num_samples

    # 输出总样本数
    print(f"总样本数：{total_samples}")


if __name__ == '__main__':
    main()