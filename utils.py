#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch 
import os 
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from contextlib import nullcontext
from diffusers import FluxControlPipeline, FluxTransformer2DModel
from diffusers.utils import load_image, make_image_grid, is_wandb_available
from diffusers.training_utils import free_memory
from accelerate.logging import get_logger


logger = get_logger(__name__)


if is_wandb_available():
    import wandb

def expand_tensor_to_dim(tensor, ndim):
    tensor = tensor.flatten()
    while len(tensor.shape) < ndim:
        tensor = tensor.unsqueeze(-1)
    return tensor

def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)


def log_validation(flux_transformer, args, accelerator, weight_dtype, step, is_final_validation=False):
    logger.info("Running validation... ")

    if not is_final_validation:
        flux_transformer = accelerator.unwrap_model(flux_transformer)
        pipeline = FluxControlPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=flux_transformer,
            torch_dtype=weight_dtype,
        )
    else:
        transformer = FluxTransformer2DModel.from_pretrained(args.output_dir, torch_dtype=weight_dtype)
        pipeline = FluxControlPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=transformer,
            torch_dtype=weight_dtype,
        )

    pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []
    if is_final_validation or torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type, weight_dtype)

    vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
    for validation_prompt, validation_image in zip(validation_prompts, validation_images):
        validation_image = load_image(validation_image)
        width, height = validation_image.size
        if height % (vae_scale_factor * 2) != 0 or width % (vae_scale_factor * 2) != 0:
            height, width = (1024, 1024)

        for gs in args.validation_guidance_scale:
            images = []
            for _ in range(args.num_validation_images):
                with autocast_ctx:
                    image = pipeline(
                        prompt=validation_prompt,
                        control_image=validation_image,
                        num_inference_steps=50,
                        guidance_scale=gs,
                        generator=generator,
                        max_sequence_length=512,
                        height=height,
                        width=width,
                    ).images[0]
                images.append(image)
            image_logs.append(
                {
                    "validation_image": validation_image,
                    "images": images, 
                    "validation_prompt": validation_prompt,
                    "guidance_scale": gs
                }
            )

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            formatted_images = []
            for log in image_logs:
                images = log["images"]
                guidance_scale = log["guidance_scale"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                formatted_images.append(wandb.Image(validation_image, caption=f"Conditioning (gs: {guidance_scale})"))
                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

        del pipeline
        free_memory()
        return image_logs


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            make_image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
# flux-control-{repo_id}

These are Control weights trained on {base_model} and [TIGER-Lab/OmniEdit-Filtered-1.2M](https://huggingface.co/datasets/TIGER-Lab/OmniEdit-Filtered-1.2M).
{img_str}

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "flux",
        "flux-diffusers",
        "text-to-image",
        "diffusers",
        "control",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def log_data_samples_to_wandb(dataloader, accelerator):
    logger.info("Logging some dataset samples.")
    formatted_src_images = []
    formatted_edited_images = []
    all_prompts = []
    for i, batch in enumerate(dataloader):
        source_images = (batch["source_pixel_values"] + 1) / 2
        edited_images = (batch["edited_pixel_values"] + 1) / 2
        prompts = batch["captions"]

        if len(formatted_src_images) > 10:
            break

        for img, edited_img, prompt in zip(source_images, edited_images, prompts):
            formatted_src_images.append(img)
            formatted_edited_images.append(edited_img)
            all_prompts.append(prompt)

    logged_artifacts = []
    for img, edited_img, prompt in zip(formatted_src_images, formatted_edited_images, all_prompts):
        logged_artifacts.append(wandb.Image(img, caption="Conditioning"))
        logged_artifacts.append(wandb.Image(edited_img, caption=prompt))

    wandb_tracker = [tracker for tracker in accelerator.trackers if tracker.name == "wandb"]
    assert wandb_tracker, "wandb couldn't be found in the trackers."
    wandb_tracker[0].log({"dataset_samples": logged_artifacts})