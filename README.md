# FluxEdit

This project tries to teach [Flux.1 Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) the task of image editing from language with the [Flux Control framework](https://github.com/huggingface/diffusers/tree/main/examples/flux-control). We use the high-quality [`TIGER-Lab/OmniEdit-Filtered-1.2M`](https://huggingface.co/datasets/TIGER-Lab/OmniEdit-Filtered-1.2M/) dataset for training. Find the fine-tuned edit model here: [`sayakpaul/FLUX.1-dev-edit-v0`](https://huggingface.co/sayakpaul/FLUX.1-dev-edit-v0).

<div align="center">
<img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux-edit-artifacts/output_slow.gif" alt="GIF"/>
</div>

>[!IMPORTANT]
> Since we don't have the official Flux Control training details available, this project should be considered experimental and we welcome contributions from the community to make it better 🤗

## Setup

Install the dependencies from [`requirements.txt`](./requirements.txt) and perform any other configuration that might be needed.

The scripts were tested using PyTorch 2.5.1 and NVIDIA GPUs (H100).

## Training

We first converted the original OmniEdit dataset into Webdataset shards using [this script](./misc/convert_to_wds.py) for efficiency. This script prepares the Webdataset shards and push them to an S3 bucket. But you can configure this as per your needs.

<details>
<summary>Training Command</summary>

```bash
export LR=1e-4
export WEIGHT_DECAY=1e-4
export GUIDANCE_SCALE=30.0
export CAPTION_DROPOUT=0.0
export LR_SCHEDULER="constant"

srun --wait=60 --kill-on-bad-exit=1 accelerate launch --config_file=./misc/accelerate_ds2.yaml train_control_flux_wds.py \
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
  --validation_image "assets/car.jpg" "assets/green_creature.jpg" "assets/norte_dam.jpg" "assets/mushroom.jpg" \
  --validation_prompt "Give this the look of a traditional Japanese woodblock print." "transform the setting to a winter scene" "Change it to look like it's in the style of an impasto painting." "turn the color of mushroom to gray" \
  --seed="0" \
  --push_to_hub

echo "END TIME: $(date)"
```

</details>

Training starts on 8 GPUs using DeepSpeed. You can configure the [`accelerate` config file](./misc/accelerate_ds2.yaml) to change that.

Refer to the [`args.py`](./args.py) to know the different kinds of configurations supported. Training was conducted on a node of 8 H100s. If you prefer using Slurm, refer to this Slurm script for scheduling training.

You can also use this [minimal version of the `train.py` script](https://github.com/huggingface/diffusers/blob/main/examples/flux-control/train_control_flux.py) with a [minimal version of the OmniEdit dataset](https://huggingface.co/datasets/sayakpaul/OmniEdit-mini) for quicker prototyping.

## Inference

```py
from diffusers import FluxControlPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
import torch 

path = "sayakpaul/FLUX.1-dev-edit-v0" # to change
edit_transformer = FluxTransformer2DModel.from_pretrained(path, torch_dtype=torch.bfloat16)
pipeline = FluxControlPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=edit_transformer, torch_dtype=torch.bfloat16
).to("cuda")

image = load_image("./assets/mushroom.jpg") # resize as needed.
print(image.size)

prompt = "turn the color of mushroom to gray"
image = pipeline(
    control_image=image,
    prompt=prompt,
    guidance_scale=30., # change this as needed.
    num_inference_steps=50, # change this as needed.
    max_sequence_length=512,
    height=image.height,
    width=image.width,
    generator=torch.manual_seed(0)
).images[0]
image.save("edited_image.png")
```

### Speeding inference with a turbo LoRA

We can speed up the inference by reducing the `num_inference_steps` to produce a nice image by using turbo LoRA like [`ByteDance/Hyper-SD`](https://hf.co/ByteDance/Hyper-SD).

Make sure to install `peft` before running the code below: `pip install -U peft`.

<details>
<summary>Code</summary>

```py
from diffusers import FluxControlPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
import torch

path = "sayakpaul/FLUX.1-dev-edit-v0" # to change
edit_transformer = FluxTransformer2DModel.from_pretrained(path, torch_dtype=torch.bfloat16)
control_pipe = FluxControlPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=edit_transformer, torch_dtype=torch.bfloat16
).to("cuda")

# load the turbo LoRA
control_pipe.load_lora_weights(
    hf_hub_download("ByteDance/Hyper-SD", "Hyper-FLUX.1-dev-8steps-lora.safetensors"), adapter_name="hyper-sd"
)
control_pipe.set_adapters(["hyper-sd"], adapter_weights=[0.125])

image = load_image("./assets/mushroom.jpg") # resize as needed.
print(image.size)

prompt = "turn the color of mushroom to gray"
image = pipeline(
    control_image=image,
    prompt=prompt,
    guidance_scale=30., # change this as needed.
    num_inference_steps=8, # change this as needed.
    max_sequence_length=512,
    height=image.height,
    width=image.width,
    generator=torch.manual_seed(0)
).images[0]
image.save("edited_image.png")
```

</details>
<br><br>
<details>
<summary>Comparison</summary>

<table align="center">
  <tr>
    <th>50 steps</th>
    <th>8 steps</th>
  </tr>
  <tr>
    <td align="center"><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux-edit-artifacts/edited_car.jpg" alt="50 steps 1" width="150"></td>
    <td align="center"><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux-edit-artifacts/edited_8steps_car.jpg" alt="8 steps 1" width="150"></td>
  </tr>
  <tr>
    <td align="center"><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux-edit-artifacts/edited_norte_dam.jpg" alt="50 steps 2" width="150"></td>
    <td align="center"><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux-edit-artifacts/edited_8steps_norte_dam.jpg" alt="8 steps 2" width="150"></td>
  </tr>
  <tr>
    <td align="center"><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux-edit-artifacts/edited_mushroom.jpg" alt="50 steps 3" width="150"></td>
    <td align="center"><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux-edit-artifacts/edited_8steps_mushroom.jpg" alt="8 steps 3" width="150"></td>
  </tr>
  <tr>
    <td align="center"><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux-edit-artifacts/edited_green_creature.jpg" alt="50 steps 4" width="150"></td>
    <td align="center"><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux-edit-artifacts/edited_8steps_green_creature.jpg" alt="8 steps 4" width="150"></td>
  </tr>
</table>


</details>

You can also choose to perform quantization if the memory requirements cannot be satisfied further w.r.t your hardware. Refer to the [Diffusers documentation](https://huggingface.co/docs/diffusers/main/en/quantization/overview) to learn more.