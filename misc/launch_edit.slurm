#!/bin/bash
#SBATCH --job-name=control-flux-omniedit
#SBATCH --nodes=1
# Set the QoS
#SBATCH --qos=normal
# set 48h for job wall time limit
#SBATCH --time=2-00:00:00
# activate the requeue option
#SBATCH --requeue
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=1999G
#SBATCH --partition=YOUR_PARTION_NAME
#SBATCH -o /PATH/%x-%j-train.out

set -xe

export NCCL_ASYNC_ERROR_HANDLING=1
export LR=1e-4
export WEIGHT_DECAY=1e-4
export GUIDANCE_SCALE=30.0
export CAPTION_DROPOUT=0.0
export LR_SCHEDULER="constant"

srun --wait=60 --kill-on-bad-exit=1 accelerate launch --config_file=accelerate_ds2.yaml train_control_flux_wds.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --output_dir="omniflux-lr_${LR}-wd_${WEIGHT_DECAY}-gs_${GUIDANCE_SCALE}-cd_${CAPTION_DROPOUT}-scheduler_${LR_SCHEDULER}-sim_flow-no8bitadam" \
  --mixed_precision="bf16" \
  --per_gpu_batch_size=4 \
  --dataloader_num_workers=4 \
  --gradient_accumulation_steps=4 \
  --quality_threshold=10.0 \
  --simplified_flow \
  --gradient_checkpointing \
  --proportion_empty_prompts=$CAPTION_DROPOUT \
  --learning_rate=$LR \
  --adam_weight_decay=$WEIGHT_DECAY \
  --guidance_scale=$GUIDANCE_SCALE \
  --validation_guidance_scale="10.,20.,30.,40." \
  --report_to="wandb" --log_dataset_samples \
  --lr_scheduler=$LR_SCHEDULER \
  --lr_warmup_steps=0 \
  --checkpointing_steps=4000 \
  --resume_from_checkpoint="latest" --checkpoints_total_limit=2 \
  --max_train_steps=20000 \
  --validation_steps=400 \
  --validation_image "car.jpg" "green_creature.jpg" "norte_dam.jpg" "mushroom.jpg" \
  --validation_prompt "Give this the look of a traditional Japanese woodblock print." "transform the setting to a winter scene" "Change it to look like it's in the style of an impasto painting." "turn the color of mushroom to gray" \
  --seed="0" \
  --push_to_hub

echo "END TIME: $(date)"
