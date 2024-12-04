import argparse
import base64
import io
import os
import warnings

import clip
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import torch
from PIL import Image
from torch.utils.data import Dataset, SequentialSampler, DataLoader


def is_valid_image(image_bytes,preprocess):
    try:
        with warnings.catch_warnings(record=True) as w:
            image_buffer = io.BytesIO(image_bytes)
            images = preprocess(Image.open(image_buffer))
            for warning in w:
                if issubclass(warning.category, UserWarning):
                    return False  # 过滤掉触发警告的图像

        return True  # 图像有效（没有异常或警告）
    except Exception:
        return False  # 过滤掉触发异常的图像

class ClueWebDataset(Dataset):
    def __init__(self, data,
                 transform=None,
                 tokenizer=None) -> None:
        super().__init__()
        self.data = data
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        example = self.data.iloc[index]
        example = example.to_dict()
        text = example['alt']
        image_buffer = io.BytesIO(example['BUFFER'])

        text_data = self._load_txt(text)
        image_data = self._load_img(image_buffer)

        sample = dict(text=text_data, image=image_data)
        return sample

    def _load_img(self, image_buffer):
        img = Image.open(image_buffer)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def _load_txt(self, text):
        if self.tokenizer is not None:
            data = self.tokenizer(text,truncate=True).squeeze()
        return data

@torch.no_grad()
def calculate_clip_similarity(dataloader, model):
    similarity_list= []
    logit_scale = model.logit_scale.exp()
    for index,batch_data in enumerate(dataloader):
        text = batch_data['text']
        text_features = forward_modality(model, text, 'txt')
        image = batch_data['image']
        image_features = forward_modality(model, image, 'img')

        # normalize features
        text_features = text_features / text_features.norm(dim=1, keepdim=True).to(torch.float32)
        image_features = image_features / image_features.norm(dim=1, keepdim=True).to(torch.float32)


        # calculate scores
        similarity_scores = torch.cosine_similarity(image_features, text_features).to('cpu').numpy()
        similarity_list.extend(list(similarity_scores))



    return similarity_list

def forward_modality(model, data, flag):
    device = next(model.parameters()).device
    if flag == 'img':
        features = model.encode_image(data.to(device))
    elif flag == 'txt':
        features = model.encode_text(data.to(device))
    else:
        raise TypeError
    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default='/data4/clip-pretrain/clue_webdata',
                        help="Path to data file")
    parser.add_argument("--output_path", type=str, default='/data1/lvyuanhuiyi/meisen/clueweb_case',
                        help="Path to data file")
    parser.add_argument("--file_num", type=int, default=0,help="Path to data file")
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size to use')
    parser.add_argument('--clip-model', type=str, default='ViT-B/32',
                        help='CLIP model to use')
    parser.add_argument('--num-workers', type=int, default=8,
                        help=('Number of processes to use for data loading. '
                              'Defaults to `min(8, num_cpus)`'))
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use. Like cuda, cuda:0 or cpu')

    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    print('Loading CLIP model: {}'.format(args.clip_model))
    model, preprocess = clip.load(args.clip_model, device=device)

    parquet_files = [file for file in os.listdir(args.input_path) if file.endswith('.parquet')]

    parquet_files.sort()

    selected_file = parquet_files[args.file_num]

    input_file_path = os.path.join(args.input_path, selected_file)

    # 第一遍异常捕获过滤
    table = pq.read_table(input_file_path)
    df = table.to_pandas()
    df['is_valid'] = df['BUFFER'].apply(lambda image_buffer: is_valid_image(image_buffer, preprocess))
    data = df[df['is_valid']]

    # 第二遍过滤
    dataset = ClueWebDataset(data,
                             transform=preprocess, tokenizer=clip.tokenize)
    data_sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, args.batch_size, sampler=data_sampler,
                            num_workers=num_workers, pin_memory=True, drop_last=False)
    print('Calculating CLIP Similarity:')
    clip_score_list = calculate_clip_similarity(dataloader, model)

    data['similarity'] = clip_score_list

    # 将过滤后的数据保存到新的Parquet文件
    filtered_table = pa.Table.from_pandas(data)
    output_file_path = os.path.join(args.output_path, selected_file)
    pq.write_table(filtered_table, output_file_path)

    num_filtered = len(df) - len(data)
    print(f"过滤掉的样本数目: {num_filtered}")
    print(f"过滤后的数据已保存到 {output_file_path}。")

    print('finish')


if __name__ == '__main__':
    main()
