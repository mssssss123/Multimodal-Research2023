import argparse
import io

import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import base64
import pyarrow as pa

from PIL import Image

import pyarrow.parquet as pq

# 禁用内存限制
pq.ParquetFile.set_memory_pool(pq.default_memory_pool())

def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--parquet_file_path", type=str, default='/data3/clueweb-anchor/raw_image_text.parquet', help="Path to data file")
    parser.add_argument("--random_seed", type=int, default=2023, help="Path to data file")
    parser.add_argument("--output_file_path", type=str, default='/data1/meisen/multi-modal/open_clip-main/dataset/filter_image_text.parquet', help="Path to data file")

    args = parser.parse_args()

    df = pd.read_parquet(args.output_file_path)
    print(f"Number of records in the combined Parquet file: {len(df)}")

    df_shuffled = df.sample(frac=1, random_state=args.random_seed)

    validation_size = 10000
    validation_set = df_shuffled[:validation_size]
    training_set = df_shuffled[validation_size:]

    # 保存成两个Parquet文件
    validation_set.to_parquet('valid.parquet', index=False)
    training_set.to_parquet('train.parquet', index=False)

    print('finish')




if __name__ == '__main__':
    main()