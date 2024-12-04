import os
import random
import torch.distributed as dist

import numpy as np
import torch

def random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def smart_tokenizer_and_embedding_resize(special_tokens_dict,tokenizer,model):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def get_grouped_params(model, wd):
    params_with_wd, params_without_wd = [], []

    def apply_decay(x):
        return "gated_cross_attn_layer" in x and "ff_gate" not in x and "attn_gate" not in x and "norm" not in x and "bias" not in x

    for n, p in model.named_parameters():
        # if p.requires_grad:
        if apply_decay(n):
            params_with_wd.append(p)
        else:
            params_without_wd.append(p)

    return [
        {"params": params_with_wd, "weight_decay": wd},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def master_print(*args, **kwargs):
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        if rank == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)

def save_checkpoint(epoch, model, args, accelerator, unwrapped_model=None, global_step=None):
    """Save a checkpoint for the model."""
    # Ensure the directory exists
    if not os.path.exists(args.external_save_dir):
        os.makedirs(args.external_save_dir)

    if unwrapped_model is None:
        unwrapped_model = accelerator.unwrap_model(model)

    # Formulate the checkpoint filename based on whether it's an epoch or global_step checkpoint
    if global_step:
        checkpoint_path = f"{args.external_save_dir}/checkpoint_steps_{global_step}.pt"
        checkpoint_dict = {
            "steps": global_step,
            "model_state_dict": get_checkpoint(unwrapped_model),
        }
    else:
        checkpoint_path = f"{args.external_save_dir}/checkpoint_{epoch}.pt"
        checkpoint_dict = {"model_state_dict": get_checkpoint(unwrapped_model)}

    # Save the checkpoint if rank is 0
    if args.rank == 0:
        print(f"Saving checkpoint to {checkpoint_path}")
        accelerator.save(checkpoint_dict, checkpoint_path)

        # Save the model's configuration
        unwrapped_model.config.save_pretrained(args.external_save_dir)

        # Remove the previous checkpoint if required
        if args.delete_previous_checkpoint:
            if global_step:
                prev_checkpoint_path = f"{args.external_save_dir}/checkpoint_step_{global_step-args.save_steps_interval}.pt"
                if os.path.exists(prev_checkpoint_path):
                    os.remove(prev_checkpoint_path)
            elif epoch > 0:
                os.remove(f"{args.external_save_dir}/checkpoint_{epoch-1}.pt")


def save_checkpoint(checkpoint_dict, save_path, is_main_process, save_function):
    """Helper function to save the checkpoint."""
    save_function(checkpoint_dict, f"{save_path}/final_weights.pt", is_main_process=is_main_process)


def save_pretrained(component, save_path, is_main_process, save_function):
    """Helper function to save pretrained components."""
    component.save_pretrained(save_path, is_main_process=is_main_process, save_function=save_function, safe_serialization=False)

def save_model_weights(model, args, accelerator, save_path, tokenizer=None):
    unwrapped_model = accelerator.unwrap_model(model)
    is_main_process = accelerator.is_main_process
    unwrapped_model.config.save_pretrained(save_path)

    if args.save_hf_model:
        save_pretrained(unwrapped_model, save_path, is_main_process, accelerator.save)
        save_pretrained(tokenizer, save_path, is_main_process, accelerator.save)

    else:
        # Save based on the distributed type
        if accelerator.distributed_type == "DEEPSPEED" and accelerator.state.deepspeed_plugin.zero_stage == 3:
            checkpoint_dict = accelerator.get_state_dict(model)
        else:
            checkpoint_dict = get_checkpoint(model=unwrapped_model)

        if accelerator.distributed_type == "DEEPSPEED" and accelerator.state.deepspeed_plugin.zero_stage == 3:
            trainable_params_name = [name for name, p in unwrapped_model.named_parameters() if p.requires_grad]
            checkpoint_dict = {k: v for k, v in checkpoint_dict.items() if k in trainable_params_name}

        save_checkpoint(checkpoint_dict, save_path, is_main_process, accelerator.save)


def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            del state_dict[name]

    return state_dict