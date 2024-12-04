import os
import csv
import json
import torch
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

# 加载CLIP模型和tokenizer
model_name = "openai/clip-vit-base-patch32"


processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

# 自定义数据集类
class ImageTextDataset(Dataset):
    def __init__(self, folder_path):
        self.data = self.read_image_text_from_folders(folder_path)
        print('--')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def read_image_text_from_folders(self, folder_path):
        data = []
        folder_list = os.listdir(folder_path)
        for folder_name in folder_list:
            folder = os.path.join(folder_path, folder_name)
            if not os.path.isdir(folder):  # 只保留文件夹类型的项目
                continue

            sub_folder_list = os.listdir(folder)

            for json_path in sub_folder_list:
                if not json_path.endswith(".json"):
                    continue
                json_path = os.path.join(folder, json_path)
                with open(json_path, 'r') as file:
                    json_data = json.load(file)
                    url = json_data["url"]
                    caption = json_data["caption"]
                    original_width = json_data["original_width"]
                    original_height = json_data["original_height"]
                    key = json_data['key']
                image_path = os.path.join(folder, f"{key}.jpg")
                # image = Image.open(image_path).convert('RGB')
                example = {
                    'image_path':image_path,
                    'url':url,
                    'caption':caption,
                    'original_width':original_width,
                    'original_height':original_height,
                    'key':key
                }
                data.append(example)
        return data

    def collect_fn(self, batch):
        key_list = []
        image_list = []
        caption_list = []
        for example in batch:
            image_path = example['image_path']
            caption = example['caption']
            key = int(example['key'])
            key_list.append(key)
            image = Image.open(image_path).convert('RGB')
            # image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_list.append(image)
            caption_list.append(caption)
        encoded_images = processor(images=image_list, return_tensors="pt", padding=True)
        encoded_texts = processor(text=caption_list, return_tensors="pt", padding=True)
        return {
            'batch_key':key_list,
            'batch_image':encoded_images,
            'batch_text':encoded_texts
        }



# 计算图像和文本之间的余弦相似度得分
def compute_similarity_scores(encoded_images, encoded_texts):
    similarity_scores = torch.cosine_similarity(encoded_images, encoded_texts)
    return similarity_scores

def main():
    # 假设图像和文本所在的文件夹路径为"data_folder"
    data_folder = "/data1/lvyuanhuiyi/meisen/clueweb_details/cluewebdemo-first"
    # a = '/data1/lvyuanhuiyi/meisen/clueweb_details/cluewebdemo-first/00001'
    # sub_folder_list = os.listdir(a)
    # json_list = []
    # for json_path in sub_folder_list:
    #     if json_path.endswith(".json"):
    #         json_list.append(json_path)
    # b = []
    # for i in json_list:
    #     parts = i.split(".")
    #     b.append(int(parts[0]))
    # b.sort()
    # c = [i for i in range(1000, 2000)]
    # sb = set(b)
    # sc = set(c)
    # d = sc - sb
    # 创建自定义数据集
    dataset = ImageTextDataset(data_folder)

    # 设置批处理大小和显卡设备
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 使用DataLoader加载数据到显卡上
    data_sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, batch_size=batch_size,sampler=data_sampler,
                             collate_fn=dataset.collect_fn,
                             drop_last=False)

    # 编码图像和文本数据
    image_embeddings_list = []
    text_embeddings_list = []
    key_list = []
    model.to(device)
    model.eval()

    # 使用进度条
    for batch in tqdm(data_loader):
        batch_key = batch['batch_key']
        batch_image = batch['batch_image']
        batch_text = batch['batch_text']
        key_list.extend(batch_key)
        with torch.no_grad():
            image_embeddings = model.get_image_features(**batch_image)
            text_embeddings = model.get_text_features(**batch_text)
            image_embeddings = image_embeddings.float()
            text_embeddings = text_embeddings.float()
            image_embeddings_list.append(image_embeddings)
            text_embeddings_list.append(text_embeddings)
    # 合并编码结果
    image_embeddings = torch.cat(image_embeddings_list, dim=0)
    text_embeddings = torch.cat(text_embeddings_list, dim=0)
    # 计算相似度得分
    similarity_scores = compute_similarity_scores(image_embeddings, text_embeddings)
    similarity_scores = similarity_scores.numpy()
    key_simliar = dict()
    for key,similar in zip(key_list,similarity_scores):
        key_simliar[key] = similar.item()

    # 保存结果到CSV文件
    output_csv_file = 'output_similarity.csv'
    with open(output_csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['key', 'url', 'caption', 'original_width', 'original_height', 'similarity'])
        data = dataset.data
        for example in data:
            key = int(example['key'])
            url = example['url']
            caption = example['caption']
            original_width = example['original_width']
            original_height = example['original_height']
            similarity = key_simliar[key]
            writer.writerow([key, url, caption, original_width, original_height, similarity])


    print('---finish----')

if __name__ == '__main__':
    main()
