import argparse
import glob
import io
import os

import pandas as pd
import pyarrow.parquet as pq
import base64
import pyarrow as pa

from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--folder_path", type=str, default='/data/public/multimodal/multimodal_data/laion_aesthetics_v2_5plus_400m_0328/parquets', help="Path to data file")
    parser.add_argument("--output_root_path", type=str, default='/home/meisen/project/dataprocess/laion_aesthetics', help="Path to data file")
    args = parser.parse_args()

    # 获取文件夹中的Parquet文件列表
    parquet_files = [file for file in os.listdir(args.folder_path) if file.endswith('.parquet')]

    # 排序文件列表，以确保它们按照文件名的字母顺序排列
    parquet_files.sort()

    # 选择前40个文件
    selected_files = parquet_files[:40]
    print('------------------------filename:-----------------')
    print(selected_files)
    print('------------------------filename:-----------------')
    # 创建一个空的DataFrame来存储选定的数据
    combined_data = pd.DataFrame()

    # 逐个读取选定的Parquet文件并将它们合并到combined_data中
    for file in tqdm(selected_files, desc="Merging Parquet files"):
        file_path = os.path.join(args.folder_path, file)
        data = pd.read_parquet(file_path)
        combined_data = pd.concat([combined_data, data], ignore_index=True)

    combined_data = combined_data.iloc[:355297]
    # 将合并后的数据保存为新的Parquet文件
    output_file_path = os.path.join(args.output_root_path, 'laion_data_demo.parquet')
    combined_data.to_parquet(output_file_path, index=False)


    print('finish')




if __name__ == '__main__':
    main()