#!/bin/bash
export JAX_PLATFORMS=tpu,cpu
export PJRT_DEVICE=TPU
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
export PROFILE_EPOCH=0
export PROFILE_STEP=3
export PROFILE_DURATION_MS=40000
export PROFILE_LOGDIR=gs://bbahl/llama-det/
python3 examples/pytorch/language-modeling/run_clm.py \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --per_device_train_batch_size 128 \
  --do_train \
  --output_dir /home/$USER/tmp/test-clm \
  --overwrite_output_dir \
  --config_name llama-2-config.json \
  --cache_dir /home/$USER/cache \
  --tokenizer_name hf-internal-testing/llama-tokenizer \
  --block_size 2048 \
  --optim adafactor \
  --save_strategy no \
  --logging_strategy no \
  --torch_dtype bfloat16 \
  --dataloader_drop_last yes \
  --flash_attention \
  --max_steps 10 \
  --gradient_accumulation_steps 4