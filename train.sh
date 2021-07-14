#!/usr/bin/env bash
./run_wav2vec2_pretrain_flax.py \
    --output_dir="./third_run" \
    --num_train_epochs="5" \
    --per_device_train_batch_size="32" \
    --per_device_eval_batch_size="32" \
    --learning_rate="5e-4" \
    --weight_decay="0.01" \
    --warmup_steps="2000" \
    --model_name_or_path="./" \
    --dataset_name="common_voice" \
    --dataset_config_name="es" \
    --preprocessing_num_workers="96" \
    --max_duration_in_seconds="10.0" \
    --adam_beta1="0.9" \
    --adam_beta2="0.98" \
    --pad_to_multiple_of="16384" \
    --validation_split_percentage="5" \
    --speech_file_column="path" \
    --dtype="bfloat16" \
    --cache_dir="./data_cache" \
    --push_to_hub
