import argparse
import glob
import io
import os

import pandas as pd
import pyarrow.parquet as pq
import base64
import pyarrow as pa

from PIL import Image


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--data_file_path", type=str, default='/data1/lvyuanhuiyi/meisen/clueweb_case/filter.parquet', help="Path to data file")
    args = parser.parse_args()


    df = pd.read_parquet(args.data_file_path)
    a = df.head(10)

    print('finish')




if __name__ == '__main__':
    main()