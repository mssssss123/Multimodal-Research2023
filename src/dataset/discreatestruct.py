import copy
import json
import os

import hydra
import torch
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from src.utils.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN, IGNORE_INDEX, DIS_IMG_TOKEN
from src.utils import conversation as conversation_lib


class DiscreateInstructDataset(Dataset):
    def __init__(self, tokenizer, data_args):
        super(DiscreateInstructDataset, self).__init__()
        self.data_args = data_args
        self.tokenizer = tokenizer
        data_path = os.path.join(self.data_args.data_path,"llava_instruct_150k.json")
        self.list_data_dict = json.load(open(data_path, "r"))
        self.transform = self.load_transform(data_args.transform_config)
        self.image_tokenizer = self.load_image_tokenizer(data_args.image_tokenizer_config)

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = os.path.join(self.data_args.data_path,"train2017")
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image_tensor = self.transform(image).to('cuda')
            with torch.no_grad():
                image_ids = self.image_tokenizer.encode_image(image_torch=image_tensor)
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args,
                image_ids.tolist(),
            )
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        return data_dict

    def load_image_tokenizer(self,config_path):
        tokenizer_cfg = OmegaConf.load(config_path)
        tokenizer = hydra.utils.instantiate(tokenizer_cfg, device='cuda', load_diffusion=False)
        return tokenizer

    def load_transform(self,config_path):
        transform_cfg = OmegaConf.load(config_path)
        transform = hydra.utils.instantiate(transform_cfg)
        return transform


def preprocess_multimodal(
    sources,
    data_args,
    image_ids,
):
    # TODO: 我不知道为什么这里会是一个二维列表
    image_ids = image_ids[0]
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = sentence['value'] + '\n' + DEFAULT_IMAGE_TOKEN
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = ''.join([DIS_IMG_TOKEN.format(int(item)) for item in image_ids])
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
            # TODO: 有点糟糕 34=32+1+1
            target[cur_idx + 2:cur_idx + tokenized_len - 34] = IGNORE_INDEX
        cur_idx += tokenized_len