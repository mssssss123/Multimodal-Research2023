nohup python eval.py  \
    --model_name ClipPythia  \
    --model_path /home/meis23/project/nips2024-pretrain/checkpoint/clip_pythia_alttext/9999step  \
    --device 0  \
    --dataset_name Flickr  \
    --batch_size 16  \
    --sample_num -1  \
    --eval_caption   \
    --exp_name clip_pythia_alttext    > clip_pythia_alttext_Flickr_10000.out  2>&1 &

