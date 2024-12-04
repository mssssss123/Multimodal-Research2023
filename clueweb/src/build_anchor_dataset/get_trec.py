import argparse
import json

import faiss
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, SequentialSampler, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pickle

def load_passage_embedding(passage_embedding_path):
    with open(passage_embedding_path, 'rb') as file:
        data = pickle.load(file)
    return data

def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size to use')
    parser.add_argument('--doc_maxlen', type=int, default=128,
                        help='maxlen')
    parser.add_argument('--qry_maxlen', type=int, default=128,
                        help='maxlen')
    parser.add_argument('--threshold', type=int, default=500,
                        help='threshold')
    parser.add_argument('--model_path', type=str, default='/data2/t5-ance',
                        help='model to use')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument("--doc_file_path", type=str, default='/data1/zhoutianshuo/multi-modal/clueweb_data/raw_image_text_new.parquet',
                        help="Path to data file")

    parser.add_argument("--qry_file_path", type=str, default='/data3/clueweb-anchor/anchor.json',
                        help="Path to data file")

    parser.add_argument("--qry_save_file", type=str, default='/data3/clueweb-anchor-filter/qry_embedding.json',
                        help="Path to data file")
    parser.add_argument("--doc_save_file", type=str, default='/data3/clueweb-anchor-filter/doc_embedding.json',
                        help="Path to data file")
    args = parser.parse_args()

    qry_emb = load_passage_embedding(args.qry_save_file)
    doc_emb = load_passage_embedding(args.doc_save_file)

    qry_emb_list = []
    qry_idx_list = []
    for q_value in qry_emb:
        q_values = list(q_value.values())
        qry_emb_list.append(q_values[0])

        q_idxs = list(q_value.keys())
        qry_idx_list.append(q_idxs[0])

    doc_emb_list = []
    doc_idx_list = []
    for d_value in doc_emb:
        d_values = list(d_value.values())
        doc_emb_list.append(d_values[0])

        d_idxs = list(d_value.keys())
        doc_idx_list.append(d_idxs[0])


    # 使用 numpy.vstack 垂直堆叠数组
    qry_numpy = np.vstack(qry_emb_list)
    doc_numpy = np.vstack(doc_emb_list)

    faiss.omp_set_num_threads(16)
    cpu_index = faiss.IndexFlatIP(768)
    #cpu_index.add(np.array(doc_emb_list, dtype=np.float32))
    #query_embeds = np.array(qry_emb_list, dtype=np.float32)

    cpu_index.add(doc_numpy)
    query_embeds = qry_numpy

    D, I = cpu_index.search(query_embeds, args.threshold)

    label_idx_list = I.tolist()

    stay_num=0
    saved_id_list = []
    for id, doc_idx in enumerate(label_idx_list):
        for j_id in doc_idx:
            if qry_idx_list[id] == doc_idx_list[j_id]:
                stay_num+=1
                saved_id_list.append(id)
                break
    print('total saved %d' % stay_num)

    saved_array = np.array(saved_id_list)


    saved_file_path = 'saved_array.npy'


    np.save(saved_file_path, saved_array)


    print(f"数组已保存到文件 {saved_file_path}")
    print("----------finish---------------")



if __name__ == '__main__':
    main()

