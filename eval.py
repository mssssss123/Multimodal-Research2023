import os
import json
import argparse
import datetime
from functools import partial

import torch
import numpy as np

from benchmark.models import get_model
from benchmark.task_datasets import dataset_class_dict
from benchmark.tasks import evaluate_VQA, evaluate_Caption


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    # models
    parser.add_argument("--model_path", type=str, default="/data2/meisen/nips2024/checkpoint/llava-discreate-test/499step")
    parser.add_argument("--model_name", type=str, default="Pythia")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=-1)

    # datasets
    parser.add_argument("--dataset_name", type=str, default="TextVQA")
    parser.add_argument("--sample_num", type=int, default=-1)
    parser.add_argument("--sample_seed", type=int, default=0)

    # result_path
    parser.add_argument("--answer_path", type=str, default="./answers")
    parser.add_argument("--exp_name", type=str, default="test")

    # eval choices
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--eval_vqa", action="store_true", help="Whether to evaluate on vqa.",default=True)
    parser.add_argument("--eval_caption", action="store_true", help="Whether to evaluate on caption.")

    args = parser.parse_args()
    return args


def sample_dataset(dataset, max_sample_num=5000, seed=0):
    if max_sample_num == -1:
        return dataset

    if len(dataset) > max_sample_num:
        np.random.seed(seed)
        random_indices = np.random.choice(
            len(dataset), max_sample_num, replace=False
        )
        dataset = torch.utils.data.Subset(dataset, random_indices)
    return dataset


def get_eval_function(args):
    if args.eval_vqa:
        eval_func = evaluate_VQA
    elif args.eval_caption:
        eval_func = evaluate_Caption
    else:
        raise NotImplementedError("Invalid choice of evaluation function")

    if args.max_new_tokens != -1:
        eval_func = partial(eval_func, max_new_tokens=args.max_new_tokens)

    if args.question is not None:
        eval_func = partial(eval_func, question=args.question)

    return eval_func


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    model = get_model(args, device=torch.device('cuda'))
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    answer_path = f"{args.answer_path}/{args.model_name}/{args.exp_name}"

    result = {}

    eval_function = get_eval_function(args)
    if eval_function is not None:
        dataset = dataset_class_dict[args.dataset_name]()
        dataset = sample_dataset(dataset, args.sample_num, args.sample_seed)
        metrics = eval_function(model, dataset, args.model_name, args.dataset_name, time, args.batch_size,
                                answer_path=answer_path)
        result[args.dataset_name] = metrics

    result_path = os.path.join(os.path.join(answer_path, time), 'result.json')
    with open(result_path, "w") as f:
        f.write(json.dumps(result, indent=4))


if __name__ == "__main__":
    args = parse_args()
    main(args)