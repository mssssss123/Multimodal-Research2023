import os
import time

import torch
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_constant_schedule_with_warmup

from src.dataset.data_utils import make_supervised_data_module
from src.model.clip_pythia.model import ClipPythiaForCausalLM
from src.utils.constants import DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN
from src.utils.distributed import world_info_from_env
from src.utils.train_args import parse_args
from src.utils.train_utils import random_seed, smart_tokenizer_and_embedding_resize, get_grouped_params, \
    AverageMeter, master_print, save_model_weights


def train_one_epoch(args, model, epoch, tokenizer, optimizer, lr_scheduler, train_dataloader, accelerator, device_id,
                    wandb):
    num_batches_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps

    model.train()

    # setup logging
    step_time_m = AverageMeter()  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = AverageMeter()  # avg time to load one batch of ClueWeb (= 1 batch regardless of gradient accum)
    end = time.time()

    for num_steps, batch_clueweb in tqdm(
            enumerate(train_dataloader),
            disable=args.rank != 0,
            total=args.total_training_steps,
            initial=(epoch * num_batches_per_epoch),
    ):
        # 避免eval之后
        model.train()

        data_time_m.update(time.time() - end)

        global_step = num_steps + epoch * num_batches_per_epoch

        input_ids = batch_clueweb['input_ids']
        labels = batch_clueweb['labels']
        attention_mask = batch_clueweb['attention_mask']
        images = batch_clueweb['images']

        with accelerator.accumulate(model):
            if num_steps == 0:
                unwrapped_model = accelerator.unwrap_model(model)
                master_print(f"model: {unwrapped_model.__class__.__name__}")
                master_print(f"model dtype: {unwrapped_model.dtype if hasattr(unwrapped_model, 'dtype') else 'None'}")

            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                images=images,
            )['loss']

            accelerator.backward(loss)

            #### BACKWARD PASS ####
            mean_loss = loss.detach().mean()
            cur_batch_max_tokens = input_ids.shape[1]

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if accelerator.sync_gradients:
                if args.rank == 0 and args.report_to_wandb:
                    # compute within rank 0
                    samples_per_second = args.gradient_accumulation_steps * args.batch_size * args.world_size / step_time_m.sum
                    samples_per_second_per_gpu = args.gradient_accumulation_steps * args.batch_size / step_time_m.sum
                    step_time_m.reset()
                    data_time_m.reset()

                    wandb.log(
                        {
                            "data_time": data_time_m.avg,
                            "step_time": step_time_m.avg,
                            "max_tokens": cur_batch_max_tokens,
                            "train_samples_per_second": samples_per_second,
                            "train_samples_per_second_per_gpu": samples_per_second_per_gpu,
                            "lr": optimizer.param_groups[0]["lr"],
                            "train_loss": mean_loss,
                            "global_step": global_step // args.gradient_accumulation_steps,
                        },
                        commit=True,
                    )

            # Log loss to console
            if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
                print(f"Step {num_steps + 1}/{num_batches_per_epoch} of epoch {epoch + 1}/{args.num_epochs} complete. Mean Loss: {mean_loss.item():.3f}")

            # Add a process on saving checkpoints during pretraining
            if ((num_steps + 1) % args.checkpointing_steps == 0) and args.rank == 0:
                save_path = os.path.join(args.external_save_dir, str(global_step) + 'step')
                save_model_weights(model=model, args=args, accelerator=accelerator,
                                        save_path=save_path, tokenizer=tokenizer)




def main():
    args = parse_args()
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    random_seed(args.seed)

    args.external_save_dir = os.path.join(args.external_save_dir, args.run_name)
    os.environ["WANDB_DIR"] = args.external_save_dir
    os.makedirs(args.external_save_dir, exist_ok=True)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )
    device_id = accelerator.device

    accelerator.print(f"Loading pretrained model from {args.model_name_or_path}")

    if args.model_name.lower() == "clip_pythia":

        model = ClipPythiaForCausalLM.from_pretrained(
            args.model_name_or_path,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            model_max_length=args.model_max_length,
            padding_side="right",
            # use_fast=False,
        )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    tokenizer.add_special_tokens({
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
    })

    if args.vision_tower is not None:
        model_vision_dict = model.get_model().initialize_vision_modules(
            vision_tower=args.vision_tower,
            mm_vision_select_layer=args.mm_vision_select_layer,
        )
        vision_config = model_vision_dict['vision_config']
        args.image_token_len = model_vision_dict['image_token_len']
        args.image_processor = model_vision_dict['image_processor']
        args.is_multimodal = True

        model.config.image_aspect_ratio = args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        if args.bits == 16:
            model.to(dtype=torch.bfloat16 if args.bf16 else torch.float16)

        model.config.mm_use_im_start_end = args.mm_use_im_start_end
        model.config.mm_projector_lr = args.mm_projector_lr
        vision_config.use_im_start_end = args.mm_use_im_start_end
        vision_config.mm_use_im_patch_token = args.mm_use_im_patch_token

        model.initialize_vision_tokenizer(args, tokenizer=tokenizer)

    model.config.freeze_backbone = args.freeze_backbone
    if args.freeze_backbone:
        # 所有参数fix
        model.requires_grad_(False)

        # 打开adapter
        model.config.tune_mm_mlp_adapter = args.tune_mm_mlp_adapter
        if args.tune_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        # 打开vision moudle
        model.config.tune_vision_tower = args.tune_vision_tower
        if args.tune_vision_tower:
            for p in model.get_model().vision_tower.parameters():
                p.requires_grad = True

    # 检查训练参数
    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    params_have_grad = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f'params_no_grad: {params_no_grad}')
    print(f'params_have_grad: {params_have_grad}')

    accelerator.wait_for_everyone()
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=args)

    print(f"Start running training on rank {args.rank}.")

    if args.rank == 0 and args.report_to_wandb:
        print(f"Logging to wandb as {args.wandb_entity}/{args.wandb_project}/{args.run_name}")
        # 离线定义
        wandb.init(
            project=args.wandb_project,
            # entity=args.wandb_entity,
            name=args.run_name,
            mode="dryrun",
        )

    train_dataloader = data_module['data_loader']
    total_training_steps = len(train_dataloader) * args.num_epochs
    resume_from_epoch = 0
    optimizer = torch.optim.AdamW(get_grouped_params(model, wd=args.weight_decay), lr=args.learning_rate)

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    args.warmup_steps = total_training_steps * args.warmup_steps_ratio if args.warmup_steps_ratio is not None else args.warmup_steps
    args.warmup_steps = args.warmup_steps // args.gradient_accumulation_steps
    args.total_training_steps = total_training_steps // args.gradient_accumulation_steps

    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.total_training_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.total_training_steps,
        )
    elif args.lr_scheduler == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    if args.rank == 0 and args.report_to_wandb:
        wandb.config.update(vars(args))

    args.distributed_type = accelerator.distributed_type

    model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(model, optimizer, lr_scheduler,
                                                                          train_dataloader)
    model.train()

    for epoch in range(resume_from_epoch, args.num_epochs):
        train_one_epoch(
            args=args,
            model=model,
            epoch=epoch,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataloader=train_dataloader,
            accelerator=accelerator,
            device_id=device_id,
            wandb=wandb,
        )
        if args.rank == 0:
            save_path = args.external_save_dir
            save_path = os.path.join(save_path, str(epoch) + 'epoch')
            save_model_weights(model=model, args=args, accelerator=accelerator,
                               save_path=save_path, tokenizer=tokenizer)
            print(f"Saved checkpoint at epoch {epoch + 1}.")
        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    if args.rank == 0:
        save_path = args.external_save_dir
        save_path = os.path.join(save_path, 'final')
        save_model_weights(model=model, args=args, accelerator=accelerator,
                           save_path=save_path, tokenizer=tokenizer)
        master_print(f"Saved Final checkpoint. Finish!")


if __name__ == "__main__":
    main()