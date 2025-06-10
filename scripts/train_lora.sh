export MODEL_NAME="EasyAnimateV5.1-7b-zh"
export DATASET_NAME=""
export DATASET_META_NAME=""
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# When train model with multi machines, use "--config_file accelerate.yaml" instead of "--mixed_precision='bf16'".
accelerate launch --main_process_port 29502 --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --config_path "config/easyanimate_video_v5.1_magvit_qwen.yaml" \
  --image_sample_size=96 \
  --video_sample_size=96 \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=11 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=1 \
  --num_train_epochs=300 \
  --checkpointing_steps=100 \
  --learning_rate=3e-04 \
  --seed=42 \
  --low_vram \
  --output_dir="" \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --adam_weight_decay=5e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --rank=16 \
  --network_alpha=8 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --loss_type="flow" \
  --use_deepspeed \
  --uniform_sampling \
  --train_mode="normal"\
  --enable_bucket \
