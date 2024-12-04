import argparse

def parse_args():
    """
    Parse the command line arguments and perform the initial setup.
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Main training script for the model")
    # Model configuration arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="path to huggingface model or model identifier from local path or huggingface.co",
        default='/data4/pythia/pythia-410m',
        # default='/data4/LLaMA-2/hf/Llama-2-7b-hf',
    )
    parser.add_argument(
        "--vision_tower",
        type=str,
        default='/data1/zhoutianshuo/pretrain-model/clip-vit-large-patch14',
    )
    parser.add_argument("--model_name", type=str, default='pythia')
    parser.add_argument("--mm_vision_select_layer", type=int, default=-1)
    parser.add_argument("--mm_use_im_start_end", action="store_true", default=True)
    parser.add_argument("--mm_use_im_patch_token", action="store_true", default=True)
    parser.add_argument("--freeze_backbone", action="store_true", default=False)
    parser.add_argument("--tune_mm_mlp_adapter", action="store_true", default=True)
    parser.add_argument("--tune_vision_tower", action="store_true", default=True)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        help="path to huggingface model or model identifier from local path or huggingface.co",
        default='/data2/meisen/nips2024/checkpoint/llava/final',
    )
    parser.add_argument("--transform_config", type=str, default='/data2/meisen/nips2024/src/model/discreate/configs/transform/clip_transform.yaml')
    parser.add_argument("--image_tokenizer_config", type=str, default='/data2/meisen/nips2024/src/model/discreate/configs/tokenizer/seed_llama_tokenizer.yaml')

    # Dataset configuration arguments
    parser.add_argument("--image_aspect_ratio", type=str, default='square')
    parser.add_argument("--data_path", type=str,
                        default='/data1/lvyuanhuiyi/meisen/test_demo',
                        # default='/data2/meisen/mm_dataset/llava',
                        )
    parser.add_argument("--dataset_name", type=str, default='clueweb_discreate')
    parser.add_argument("--text_type", type=str, default='surround')
    parser.add_argument("--is_multimodal", action="store_true", default=True)
    parser.add_argument("--data_file_num", type=int, default=1)
    parser.add_argument("--image_location", type=str, default='right')

    # Training configuration arguments
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--external_save_dir",
        type=str,
        default='/data2/meisen/nips2024/checkpoint',
        help="set to save model to external path",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--bits", type=int, default=32)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--mm_projector_lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--run_name",
        type=str,
        default="llava-discreate-test",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument("--report_to_wandb", default=True, action="store_true")
    parser.add_argument("--wandb_project", type=str, default='llava-discreate')
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument(
        "--lr_scheduler",
        default="linear",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--warmup_steps_ratio", default=None, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--logging_steps", type=int, default=100, help="log loss every n steps")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="checkpointing every n steps",
    )
    parser.add_argument("--save_hf_model", default=True, action="store_true")


    args = parser.parse_args()

    return args
