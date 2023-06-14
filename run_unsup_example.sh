#!/bin/bash


python train.py \
    --model_name_or_path roberta-base \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/my-unsup-simcse-robert-base-focal \
    --num_train_epochs 1 \
    --per_device_train_batch_size 512 \
    --learning_rate 1e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --m 0.3\
    --do_train \
    --do_eval \
    --fp16 \
    "$@"