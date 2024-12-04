import argparse
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from pyarrow import default_memory_pool
from PIL import Image


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--parquet_file_path", type=str,
                        default='/home/meisen/project/dataprocess/laion_aesthetics/laion_data_demo.parquet',
                        help="Path to data file")
    parser.add_argument("--random_seed", type=int, default=2023, help="Path to data file")
    args = parser.parse_args()

    # 配置内存池大小
    large_memory_pool = default_memory_pool().memory_map_bytes(1024 * 1024 * 1024 * 50)  # 10 GB

    with default_memory_pool(large_memory_pool):
        df = pd.read_parquet(args.parquet_file_path)
        print(f"Number of records in the combined Parquet file: {len(df)}")

        df_shuffled = df.sample(frac=1, random_state=args.random_seed)

        validation_size = 355297
        validation_set = df_shuffled[:validation_size]

        # 保存成两个Parquet文件
        validation_set.to_parquet('train.parquet', index=False)

    print('finish')


if __name__ == '__main__':
    main()