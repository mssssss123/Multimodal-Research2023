"""
布隆过滤器被用于进行 URL 和 alt-text 的重复项识别和去除
"""
import argparse
import os
import sys
from functools import partial
import csv
import pandas as pd
from pybloom_live import ScalableBloomFilter
from tqdm import tqdm
import multiprocessing

csv.field_size_limit(sys.maxsize)

def process_file(file_path, bloom_filter):
    filtered_data = []
    dtype_mapping = {'url': str, 'alt-text': str}
    input_data = pd.read_csv(file_path, dtype=dtype_mapping)
    for index, row in input_data.iterrows():
        url = str(row['url'])
        alt_text = str(row['text'])

        if url in bloom_filter or alt_text in bloom_filter:
            continue
        else:
            bloom_filter.add(url)
            bloom_filter.add(alt_text)
            filtered_data.append({'url': url, 'text': alt_text})
    return filtered_data

def filter_duplicates(input_csv_path, bloom_filter):
    filtered_data = process_file(input_csv_path, bloom_filter)
    return filtered_data

def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--input_root_path", type=str,
                        default='/data1/lvyuanhuiyi/meisen/clueweb_details/src/alt_text_baseline/firstFilter')
    parser.add_argument("--output_root_path", type=str,
                        default='/data1/lvyuanhuiyi/meisen/clueweb_details/src/alt_text_baseline/secondFilter')
    args = parser.parse_args()
    os.makedirs(args.output_root_path, exist_ok=True)

    csv_file_num = len([file for file in os.listdir(args.input_root_path) if file.endswith('.csv')])

    bloom_filter = ScalableBloomFilter(initial_capacity=400000000, error_rate=0.001)

    files_to_process = [os.path.join(args.input_root_path, f'first_filter_img_alt_text_{i}.csv') for i in range(csv_file_num)]

    num_processes = multiprocessing.cpu_count() // 2  # Use all available cores

    with multiprocessing.Pool(num_processes) as pool:
        func_partial = partial(filter_duplicates, bloom_filter=bloom_filter)
        results = list(tqdm(pool.imap(func_partial, files_to_process), total=len(files_to_process)))

    merged_results = []
    for result in results:
        merged_results.extend(result)

    merged_df = pd.DataFrame(merged_results)
    print("num after filter:", merged_df.shape[0])
    output_csv_file = os.path.join(args.output_root_path, 'second_filter_img_alt_text.csv')
    merged_df.to_csv(output_csv_file, index=False)
    print('------------finish----------------')

if __name__ == '__main__':
    main()
