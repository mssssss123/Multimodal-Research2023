import argparse
import json

import faiss
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, SequentialSampler, DataLoader
from transformers import AutoTokenizer, AutoModel


class DocumentDataset(Dataset):
    def __init__(self, data,tokenizer,args):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.args = args
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        example = self.data.iloc[index]
        example = example.to_dict()
        text = example['alt']
        return text

    def collect_fn(self, data):
        outputs = self.tokenizer.batch_encode_plus(
            data,
            max_length=self.args.doc_maxlen,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True,
        )
        input_ids = outputs["input_ids"]
        attention_mask = outputs["attention_mask"]
        return {
                    "doc_ids": input_ids,
                    "doc_masks": attention_mask,
                }

class QueryDataset(Dataset):
    def __init__(self, data, tokenizer, args):
        super().__init__()
        self.data = self.process_data(data)
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        example = self.data.iloc[index]
        example = example.to_dict()
        text = example['alt']
        return text
    def process_data(self,data):
        all_data = []
        for example in data:
            anchor_list = example['anchors']
            cw22id = example['ClueWeb22-ID']
            for anchor in anchor_list:
                query_text = anchor[2]
                all_data.append({'query':query_text,'label':cw22id})
        return all_data

    def collect_fn(self, data):
        batch_qry = []

        batch_target = []
        for example in data:
            label = example['label']
            query_text = example['query']
            batch_target.append(label)
            batch_qry.append(query_text)
        outputs = self.tokenizer.batch_encode_plus(
            batch_qry,
            max_length=self.args.qry_maxlen,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True,
        )
        input_ids = outputs["input_ids"]
        attention_mask = outputs["attention_mask"]
        return {
            "qry_ids": input_ids,
            "qry_masks": attention_mask,
            "label":batch_target,
        }


def retrieve(model, qry_dataloader, doc_dataloader, device, args):
    model.eval()
    model = model.module if hasattr(model, "module") else model
    doc_emb_list = []
    qry_emb_list = []
    target_list = []
    with torch.no_grad():
        for i, batch in enumerate(doc_dataloader):
            doc_inputs = batch["doc_ids"].to(device)
            doc_masks = batch["doc_masks"].to(device)
            _,doc_emb = model(doc_inputs, doc_masks)
            doc_emb_list.append(doc_emb.cpu().numpy())
        doc_emb_list = np.concatenate(doc_emb_list, 0)
        for i, batch in enumerate(qry_dataloader):
            qry_inputs = batch["qry_ids"].to(device)
            qry_masks = batch["qry_masks"].to(device)
            batch_target = batch["label"]
            _, qry_emb = model(qry_inputs, qry_masks)
            qry_emb_list.append(qry_emb.cpu().numpy())
            target_list.extend(batch_target)
        qry_emb_list = np.concatenate(qry_emb_list, 0)
        faiss.omp_set_num_threads(16)
        cpu_index = faiss.IndexFlatIP(768)
        cpu_index.add(np.array(doc_emb_list, dtype=np.float32))
        query_embeds = np.array(qry_emb_list, dtype=np.float32)
        D, I = cpu_index.search(query_embeds, max(args.threshold))
        # TODO:判断是否在，并将结果保存起来，最好直接构建个新的query数据集文件


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size to use')
    parser.add_argument('--doc_maxlen', type=int, default=128,
                        help='maxlen')
    parser.add_argument('--qry_maxlen', type=int, default=128,
                        help='maxlen')
    parser.add_argument('--threshold', type=int, default=1000,
                        help='threshold')
    parser.add_argument('--model_path', type=str, default='/data1/meisen/pretrained_model/t5-base',
                        help='model to use')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument("--doc_file_path", type=str, default='/data3/clueweb-anchor/raw_image_text.parquet',
                        help="Path to data file")
    parser.add_argument("--qry_file_path", type=str, default='/data3/clueweb-anchor/anchor.json',
                        help="Path to data file")
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path)
    model.to(device)

    # 加载document
    doc_data = pd.read_parquet(args.doc_file_path)

    doc_dataset = DocumentDataset(doc_data, tokenizer, args)
    doc_sampler = SequentialSampler(doc_dataset)
    doc_dataloader = DataLoader(
        doc_dataset,
        sampler=doc_sampler,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=doc_dataset.collect_fn
    )

    # 加载query
    with open(args.qry_file_path, 'r') as json_file:
        qry_data = json.load(json_file)
    qry_dataset = QueryDataset(qry_data, tokenizer, args)
    qry_sampler = SequentialSampler(qry_dataset)
    qry_dataloader = DataLoader(
        qry_dataset,
        sampler=qry_sampler,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=qry_dataset.collect_fn
    )

    retrieve(model, qry_dataloader, doc_dataloader, device,args)





if __name__ == '__main__':
    main()