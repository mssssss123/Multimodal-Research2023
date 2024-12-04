"""

"""
import argparse
import json
import os.path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq





if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument("--file_num", type=int, default=0)
    parser.add_argument("--input_root_path", type=str, default='/data3/meisen/clueweb-imgtext/firstfilter')
    parser.add_argument("--output_root_path", type=str, default='/data3/meisen/clueweb-imgtext/secondfilter')
    args = parser.parse_args()
    os.makedirs(args.output_root_path, exist_ok=True)
    input_file = f'first_filter_img_text_{args.file_num}.parquet'
    input_file_path = os.path.join(args.input_root_path, input_file)
    print('------------start----------------')
    table = pq.read_table(input_file_path)
    df = table.to_pandas()
    print("num before filter:", df.shape[0])
    df_no_duplicates = df.drop_duplicates(subset=['src'], keep='first')
    table_no_duplicates = pa.Table.from_pandas(df_no_duplicates)
    output_file = f'second_filter_img_text_{args.file_num}.parquet'
    output_file_path = os.path.join(args.output_root_path, output_file)
    pq.write_table(table_no_duplicates, output_file_path)
    print("num after filter:", df_no_duplicates.shape[0])






    print('------------finish----------------')





