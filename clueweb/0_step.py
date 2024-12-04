import argparse
import csv
import multiprocessing


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

def generate_cw22id_list(cw22_filenum_dict):
    cw22id_list = []
    for k,v in cw22_filenum_dict.items():
        prefix = k
        count = int(v)
        for j in range(count):
            cw22id = f"clueweb22-{prefix}-{j:05d}"
            cw22id_list.append(cw22id)
    return cw22id_list

def chunk_list(lst, num_chunks):
    avg_chunk_size = len(lst) // num_chunks
    chunks = [lst[i:i + avg_chunk_size] for i in range(0, len(lst), avg_chunk_size)]
    return chunks

def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--file_num", type=int, default=0)
    parser.add_argument("--output_path", type=str, default='/data3/meisen/clueweb-imgtext/raw')
    parser.add_argument("--csv_file_path", type=str, default='/data4/ClueWeb22/record_counts/html/en00_counts.csv')
    parser.add_argument("--root_path", type=str, default='/data4/ClueWeb22')
    args = parser.parse_args()

    # 读取record，得到每个clueweb文档下子文档的数量
    cw22_filenum_dict = read_csv_to_dict(args.csv_file_path)
    # 将字典按照100进行分片，得到字典列表。e.g: 字典长度4700-> 4700/100=47个存储100个原字典元素的字典
    cw22_filenum_dict_list = chunk_dict(cw22_filenum_dict, 100)
    # 该分片下的所有clueweb网页id
    cw22id_list = generate_cw22id_list(cw22_filenum_dict_list[args.file_num])

    print('--------load_files--------------')
    print(len(cw22id_list))
    print('--------load_files_finished--------------')

    manager = multiprocessing.Manager()
    results = manager.list()

    # 使用cpu线程数量
    print(f'you have {multiprocessing.cpu_count()} cpu threads!')
    # num_processes = multiprocessing.cpu_count()
    num_processes = 30
    chunks = chunk_list(cw22id_list, num_processes)
    print('---')













if __name__ == '__main__':
    main()