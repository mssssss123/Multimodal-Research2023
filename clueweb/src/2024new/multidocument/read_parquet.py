import argparse
import glob
import io
import os

import pandas as pd
import pyarrow.parquet as pq
import base64
import pyarrow as pa

from PIL import Image
from io import BytesIO


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--data_file_path", type=str, default='/data4/clip-pretrain/clue_webdata/000e71358b5152dda801d0982e14b507.parquet', help="Path to data file")
    args = parser.parse_args()


    df = pd.read_parquet(args.data_file_path)
    column_name = 'cw22id'
    # df_sorted = df.sort_values(by=column_name)
    # a = df_sorted.head(100)

    duplicates = df[df.duplicated(subset=[column_name], keep=False)]
    df_sorted = duplicates.sort_values(by=column_name)

    grouped_data = df_sorted.groupby(column_name).apply(lambda x: x.to_dict(orient='records')).to_dict()
    a = grouped_data['clueweb22-en0001-00-00418']

    # 保存第一个图片
    image1 = a[0]['BUFFER']
    image1_bytes_io = BytesIO(image1)
    image1_image = Image.open(image1_bytes_io)
    image1_image.save('image1.png')

    # 保存第二个图片
    image2 = a[1]['BUFFER']
    image2_bytes_io = BytesIO(image2)
    image2_image = Image.open(image2_bytes_io)
    image2_image.save('image2.png')

    print('finish')




if __name__ == '__main__':
    main()