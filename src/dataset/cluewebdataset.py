import copy
import os
import re

import pandas as pd

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX, \
    DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN
from src.utils import conversation as conversation_lib

class ClueWebDataset(Dataset):
    def __init__(self, tokenizer, data_args):
        super(ClueWebDataset, self).__init__()
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.data = self.load_data(data_args.data_path)

    def load_data(self, input_filename):
        if os.path.isfile(input_filename):
            data = pd.read_parquet(input_filename)

        elif os.path.isdir(input_filename):
            parquet_files = [os.path.join(input_filename, file) for file in os.listdir(input_filename) if
                             file.endswith('.parquet')]
            parquet_files.sort()
            # 不使用全部数据
            if self.data_args.data_file_num != -1:
                parquet_files = parquet_files[:self.data_args.data_file_num]
            data_frames = []
            for file_path in parquet_files:
                data_frames.append(pd.read_parquet(file_path))
            data = pd.concat(data_frames, ignore_index=True)
        return data

    def textpreprocess(self, question):

        question = re.sub(r'[^\w\s.,?!()"\']', "", question)
        question = question.strip(" ")
        question = re.sub(r"\s{2,}", " ", question)
        question = question.lstrip("\n")
        question = question.rstrip("\n")
        question = question.strip(" ").strip("\n")

        return question

    def __str__(self):
        return f"type: {type(self)}, length: {len(self)}"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data.iloc[i]
        # 处理图像
        processor = self.data_args.image_processor
        image_path = example['IMAGE_PATH']
        try:
            img = Image.open(image_path).convert('RGB')
            # TODO: 某些样本图像这里会导致报错,暂时使用这种方式规避掉
            img = processor.preprocess(img, return_tensors='pt')['pixel_values'][0]
        except Exception as e:
            print(f"Error occurred while processing image: {e}")
            return self.__getitem__(i + 1)
            # 另一种可能的解决办法
            # fake_example = self.data.iloc[0]
            # image_path = fake_example['IMAGE_PATH']
            # img = Image.open(image_path).convert('RGB')
            # img = processor.preprocess(img, return_tensors='pt')['pixel_values'][0]

        # 处理文本
        text = example[self.data_args.text_type]
        text = self.textpreprocess(text)
        sources = []
        a = {'from':'human','value':'Provide a description of the given image.\n<image>'}
        b = {'from':'gpt','value':text}
        sources.append(a)
        sources.append(b)
        sources = [sources]
        # 得到input_ids和label
        sources = preprocess_multimodal(
            sources,
            self.data_args)
        data_dict = preprocess(
            sources,
            self.tokenizer,
            )

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if image_path is not None:
            data_dict['image'] = img
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict



def preprocess_multimodal(
    sources,
    data_args,
):
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    image_token_len = data_args.image_token_len
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                if data_args.image_location == 'left':
                    sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                elif data_args.image_location == 'right':
                    sentence['value'] = sentence['value'] + '\n' + DEFAULT_IMAGE_TOKEN
                elif data_args.image_location == 'origin':
                    print('not finished')
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')

            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)


    return sources

def preprocess(
    sources,
    tokenizer,
):
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        # TODO: pythia需要额外添加bos token和 eos token
        conversation = DEFAULT_BOS_TOKEN + conversation + DEFAULT_EOS_TOKEN
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        # TODO: pythia需要额外添加bos token和eos token
        tokenized_lens = _tokenize_fn([DEFAULT_BOS_TOKEN + header] + [s["value"] for s in source[:-1]] + [source[-1]["value"] + DEFAULT_EOS_TOKEN],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def _tokenize_fn(strings,
                 tokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]

    # TODO: 保证input_ids和label以eos token
    for idx, ii in enumerate(input_ids):
        if ii[-1] != tokenizer.eos_token_id:
            input_ids[idx][-1] = tokenizer.eos_token_id
            labels[idx][-1] = tokenizer.eos_token_id

    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len