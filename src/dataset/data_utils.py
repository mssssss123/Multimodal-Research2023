from typing import Dict, Sequence
from dataclasses import dataclass, field

import torch
import transformers
from torch.utils.data import RandomSampler, DataLoader

from src.utils.constants import IGNORE_INDEX


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if data_args.dataset_name == 'clueweb':
        from src.dataset.cluewebdataset import ClueWebDataset
        dataset_cls = ClueWebDataset
    elif data_args.dataset_name == 'llava_instruct':
        from src.dataset.llavainstruct import LlavaInstructDataset
        dataset_cls = LlavaInstructDataset
    elif data_args.dataset_name == 'clueweb_discreate':
        from src.dataset.cluewebdataset_discreate import DiscreateClueWebDataset
        dataset_cls = DiscreateClueWebDataset
    elif data_args.dataset_name == 'discreate_instruct':
        from src.dataset.discreatestruct import DiscreateInstructDataset
        dataset_cls = DiscreateInstructDataset


    train_dataset = dataset_cls(
        tokenizer=tokenizer,
        data_args=data_args
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    sampler = RandomSampler(train_dataset, replacement=True, num_samples=len(train_dataset))
    dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=data_args.batch_size,
        num_workers=data_args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=data_collator,
    )
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator,
                data_loader=dataloader,)