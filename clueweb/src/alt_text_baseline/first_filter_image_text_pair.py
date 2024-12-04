"""
对image_url端处理：只保留jpg/png/jpeg，并排除包含tlogo, button, icon, plugin, widget的url
对alt-text处理:排除空/No alt attribute，长度小于5
"""
import argparse
import csv
import multiprocessing
import os.path
import sys

import numpy as np
import pandas as pd

csv.field_size_limit(sys.maxsize)

def filter_img_alt_text(input_data):
    filtered_data = []
    for index, row in input_data.iterrows():
        image_url = str(row['src'])
        alt_text = str(row['alt'])
        if image_url.endswith(('.jpg', '.png', '.jpeg')) and \
                not any(keyword in image_url for keyword in ['logo', 'button', 'icon', 'plugin', 'widget']):
            alt_text_cleaned = alt_text.strip().lower()
            if alt_text_cleaned != '' and alt_text_cleaned != 'no alt attribute' and len(alt_text_cleaned) >= 5:
                filtered_data.append({'url': image_url, 'text': alt_text})
    filtered_df = pd.DataFrame(filtered_data)
    return filtered_df


def filter_img_alt_text_parallel(data):
    num_processes = max(1, multiprocessing.cpu_count() // 2)  # 使用一半的核心数
    data_chunks = np.array_split(data, num_processes)

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(filter_img_alt_text, data_chunks)

    return pd.concat(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument("--file_num", type=int, default=0)
    parser.add_argument("--input_root_path", type=str, default='/data1/lvyuanhuiyi/meisen/clueweb_details/src/alt_text_baseline')
    parser.add_argument("--output_root_path", type=str, default='/data1/lvyuanhuiyi/meisen/clueweb_details/src/alt_text_baseline/firstFilter')
    args = parser.parse_args()
    os.makedirs(args.output_root_path, exist_ok=True)

    input_csv_file = f'raw_img_alt_text_{args.file_num}.csv'
    input_file_path = os.path.join(args.input_root_path,input_csv_file)

    # use for debug
    input_data = pd.read_csv(input_file_path,header=0,nrows=100)

    # input_data = pd.read_csv(input_file_path)
    print("num before filter:", input_data.shape[0])
    filter_data = filter_img_alt_text_parallel(input_data)
    # filter_data = filter_img_alt_text(input_data)
    print("num after filter:", filter_data.shape[0])

    output_csv_file = f'first_filter_img_alt_text_{args.file_num}.csv'
    output_file_path = os.path.join(args.output_root_path,output_csv_file)
    filter_data.to_csv(output_file_path, index=False)
    print('------------finish----------------')





