"""
该代码读取ClueWeb22 Raw HTML,提取IMG标签内容，保存img src地址和alt-text
由于直接提取，所以存在着alt-text不存在等等问题，是一个非常粗的版本
TODO:如何将surrounding-text考虑进来呢
"""
import argparse
import csv
import multiprocessing

from bs4 import BeautifulSoup
from ClueWeb22Api import ClueWeb22Api
from tqdm import tqdm  # 添加进度条


def generate_cw22id_list(cw22_filenum_dict):
    cw22id_list = []
    for k,v in cw22_filenum_dict.items():
        prefix = k
        count = int(v)
        for j in range(count):
            cw22id = f"clueweb22-{prefix}-{j:05d}"
            cw22id_list.append(cw22id)


    return cw22id_list

def chunk_dict(input_dict, chunk_size):
    keys = list(input_dict.keys())
    num_chunks = len(keys) // chunk_size + (1 if len(keys) % chunk_size != 0 else 0)

    chunked_dicts = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk_keys = keys[start_idx:end_idx]
        chunk = {key: input_dict[key] for key in chunk_keys}
        chunked_dicts.append(chunk)

    return chunked_dicts

def read_csv_to_dict(csv_file):
    data_dict = {}
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 2:
                key = row[0]
                value = row[1]
                data_dict[key] = value
    return data_dict



def get_image_and_altText(cw22id, root_path):
    img_text_list = []
    try:
        clueweb_api = ClueWeb22Api(cw22id, root_path)
        html_string = clueweb_api.get_html_from_warc()
        soup = BeautifulSoup(html_string, 'html.parser')

        img_tags = soup.find_all('img')
        for img in img_tags:
            alt = img.get('alt', 'No alt attribute')
            src = img.get('src', 'No src attribute')
            img_text_list.append({'src': src, 'alt': alt})
    except Exception as e:
        print(f"Error processing {cw22id}: {str(e)}")
    return img_text_list


def process_chunk(chunk, root_path, results, process_id, total_processes):
    pbar = tqdm(chunk, position=process_id, desc=f"Process {process_id + 1}/{total_processes}")

    local_results = []  # 用于在每个进程中收集结果

    for cw22id in pbar:
        cur_list = get_image_and_altText(cw22id, root_path)
        local_results.extend(cur_list)

    results.extend(local_results)  # 将本地结果添加到共享的结果列表中
    pbar.close()

def chunk_list(lst, num_chunks):
    avg_chunk_size = len(lst) // num_chunks
    chunks = [lst[i:i + avg_chunk_size] for i in range(0, len(lst), avg_chunk_size)]
    return chunks

def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--file_num", type=int, default=0)
    args = parser.parse_args()

    csv_file_path = '/data2/clueweb/record_counts/html/en00_counts.csv'
    cw22_filenum_dict = read_csv_to_dict(csv_file_path)
    cw22_filenum_dict_list = chunk_dict(cw22_filenum_dict, 100)
    cw22id_list = generate_cw22id_list(cw22_filenum_dict_list[args.file_num])

    print('--------load_files--------------')
    print(len(cw22id_list))
    print('--------load_files_finished--------------')

    root_path = '/data2/clueweb'
    manager = multiprocessing.Manager()
    results = manager.list()

    num_processes = multiprocessing.cpu_count()  # 使用所有可用的CPU核心
    chunks = chunk_list(cw22id_list, num_processes)
    processes = []

    for process_id, chunk in enumerate(chunks):
        p = multiprocessing.Process(target=process_chunk, args=(chunk, root_path, results, process_id, num_processes))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print('--------get_data--------------')
    print(len(results))
    print('--------get_data_finished--------------')

    # 将提取的图片链接和alt文本保存到CSV文件中
    output_csv_file = f'raw_img_alt_text_{args.file_num}.csv'
    with open(output_csv_file, 'w', newline='') as csvfile:
        fieldnames = ['src', 'alt']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print('--------all_finished--------------')


if __name__ == '__main__':
    main()