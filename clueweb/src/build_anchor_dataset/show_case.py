import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--image_file_path", type=str, default='/data1/zhoutianshuo/multi-modal/clueweb_data/datasets_new/image.parquet', help="Path to data file")
    parser.add_argument("--text_file_path", type=str, default='/data1/zhoutianshuo/multi-modal/clueweb_data/datasets_new/text.parquet', help="Path to data file")
    parser.add_argument("--query_file_path", type=str, default='/data1/zhoutianshuo/multi-modal/clueweb_data/datasets_new/test.parquet', help="Path to data file")
    args = parser.parse_args()

    input_data = pd.read_parquet(args.query_file_path)
    for data in input_data.itertuples():
        print('----')












if __name__ == '__main__':
    main()
