CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup accelerate launch --config_file=/home/meis23/project/nips2024-pretrain/script/accelerate_config.yaml \
    instruct_tune.py \
    --pretrained_model_name_or_path=/home/meis23/project/nips2024-pretrain/checkpoint/clip_pythia_alttext/9999step \
    --vision_tower=/home/meis23/project/pretrained_model/clip-vit-large-patch14 \
    --model_name=clip_pythia \
    --mm_use_im_start_end \
    --mm_use_im_patch_token \
    --tune_mm_mlp_adapter \
    --tune_vision_tower \
    --data_path=/home/meis23/project/mmdataset/llava \
    --dataset_name=llava_instruct \
    --is_multimodal \
    --seed=2024 \
    --external_save_dir=/home/meis23/project/nips2024-pretrain/checkpoint/instruct \
    --gradient_accumulation_steps=1 \
    --model_max_length=512 \
    --bits=32 \
    --batch_size=8 \
    --run_name=clip_pythia_alt_instruct \
    --report_to_wandb \
    --wandb_project=clip_pythia_alt_instruct \
    --num_epochs=3 \
    --learning_rate=5e-5 \
    --lr_scheduler=cosine \
    --warmup_steps_ratio=0.03 \
    --weight_decay=0. \
    --logging_steps=10000 \
    --checkpointing_steps=10000 \
    --save_hf_model  > clip_pythia_alt_instruct.out  2>&1 &