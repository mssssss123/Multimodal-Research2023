"""
对image_url端处理：只保留jpg/png/jpeg，并排除包含tlogo, button, icon, plugin, widget的url
对alt-text处理:排除空/No alt attribute，长度小于5
"""
import argparse
import json
import os.path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def filter_img_alt_text(input_data):
    filter_data = []
    for data in input_data:
        if 'surround' not in data or len(data['surround']) == 0:
            continue
        src = str(data['src'])
        keywords_to_ignore = ['logo', 'button', 'icon', 'plugin', 'widget']
        if not src.endswith(('.jpg', '.png', '.jpeg')) or any(
                keyword.lower() in src.lower() for keyword in keywords_to_ignore):
            continue
        alt = str(data['alt'])
        alt_text_cleaned = alt.strip().lower()
        if alt_text_cleaned == '' or alt_text_cleaned == 'no alt attribute' or len(alt_text_cleaned) < 5:
            continue
        surround_list = data['surround']
        # new_surround_list = []
        # for text in surround_list:
        #     text = str(text).strip().lower()
        #     if len(text) >= 5:
        #         new_surround_list.append(text)
        # if len(new_surround_list) == 0:
        #     continue
        surround_string = " ".join(surround_list)
        nodeid = int(data['nodeid'])
        cw22id = str(data['cw22id'])
        filter_data.append({'src': src, 'alt': alt, 'nodeid': nodeid, 'cw22id': cw22id, 'surround': surround_string})
    filter_df = pd.DataFrame(filter_data)

    return filter_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument("--file_num", type=int, default=0)
    parser.add_argument("--input_root_path", type=str, default='/data3/meisen/clueweb-imgtext/raw')
    parser.add_argument("--output_root_path", type=str, default='/data3/meisen/clueweb-imgtext/firstfilter')
    args = parser.parse_args()
    os.makedirs(args.output_root_path, exist_ok=True)
    input_file = f'raw_img_text_{args.file_num}.json'
    input_file_path = os.path.join(args.input_root_path, input_file)
    print('------------start----------------')
    with open(input_file_path, 'r') as json_file:
        input_data = json.load(json_file)
    filter_data = filter_img_alt_text(input_data)


    print("num after filter:", filter_data.shape[0])


    output_file = f'first_filter_img_text_{args.file_num}.parquet'
    output_file_path = os.path.join(args.output_root_path, output_file)
    table = pa.Table.from_pandas(filter_data)
    pq.write_table(table, output_file_path)
    # with open(output_file_path, 'w') as json_outfile:
    #     json.dump(filter_data, json_outfile)



    print('------------finish----------------')





