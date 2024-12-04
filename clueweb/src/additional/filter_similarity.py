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
    parser.add_argument("--output_path", type=str, default='/data1/lvyuanhuiyi/meisen/clueweb_case',
                        help="Path to data file")
    parser.add_argument("--file_num", type=int, default=0,help="Path to data file")
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Batch size to use')
    args = parser.parse_args()

    parquet_files = [file for file in os.listdir(args.input_path) if file.endswith('.parquet')]

    parquet_files.sort()

    selected_file = parquet_files[args.file_num]

    input_file_path = os.path.join(args.input_path, selected_file)
    # input_file_path = '/data1/lvyuanhuiyi/meisen/clueweb_case/00081df3bb6184b31377e435c739b4bd.parquet'
    table = pq.read_table(input_file_path)
    df = table.to_pandas()
    filtered_df = df[df['similarity'] >= args.threshold]

    # 将过滤后的数据保存到新的Parquet文件
    filtered_table = pa.Table.from_pandas(filtered_df)
    output_file_path = os.path.join(args.output_path, selected_file)
    # output_file_path = os.path.join(args.output_path, 'filter.parquet')
    pq.write_table(filtered_table, output_file_path)

    num_filtered = len(df) - len(filtered_df)
    print(f"过滤掉的样本数目: {num_filtered}")
    print(f"过滤后的数据已保存到 {output_file_path}。")

    print('finish')

if __name__ == '__main__':
    main()