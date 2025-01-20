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

import copy
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import torch
import transformers
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxControlPipeline, FluxTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from dataset import ControlFluxDataset
from args import parse_args
from utils import encode_images, log_validation, save_model_card, expand_tensor_to_dim, log_data_samples_to_wandb

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.0.dev0")

logger = get_logger(__name__)


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_out_dir = Path(args.output_dir, args.logging_dir)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(logging_out_dir))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        AcceleratorState().deepspeed_plugin.deepspeed_config[
            "train_micro_batch_size_per_gpu"
        ] = args.per_gpu_batch_size

    # Disable AMP for MPS. A technique for accelerating machine learning computations on iOS and macOS devices.
    if torch.backends.mps.is_available():
        logger.info("MPS is enabled. Disabling AMP.")
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # DEBUG, INFO, WARNING, ERROR, CRITICAL
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load models. We will load the text encoders later in a pipeline to compute
    # embeddings.
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    flux_transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
    )
    logger.info("All models loaded successfully")

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    if not args.only_target_transformer_blocks:
        flux_transformer.requires_grad_(True)
    vae.requires_grad_(False)

    # cast down and move to the CPU
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # let's not move the VAE to the GPU yet.
    vae.to(dtype=torch.float32)  # keep the VAE in float32.
    if not args.offload:
        vae.to(accelerator.device)

    # enable image inputs
    with torch.no_grad():
        initial_input_channels = flux_transformer.config.in_channels
        new_linear = torch.nn.Linear(
            flux_transformer.x_embedder.in_features * 2,
            flux_transformer.x_embedder.out_features,
            bias=flux_transformer.x_embedder.bias is not None,
            dtype=flux_transformer.dtype,
            device=flux_transformer.device,
        )
        new_linear.weight.zero_()
        new_linear.weight[:, :initial_input_channels].copy_(flux_transformer.x_embedder.weight)
        if flux_transformer.x_embedder.bias is not None:
            new_linear.bias.copy_(flux_transformer.x_embedder.bias)
        flux_transformer.x_embedder = new_linear

    assert torch.all(flux_transformer.x_embedder.weight[:, initial_input_channels:].data == 0)
    flux_transformer.register_to_config(in_channels=initial_input_channels * 2, out_channels=initial_input_channels)

    if args.only_target_transformer_blocks:
        flux_transformer.x_embedder.requires_grad_(True)
        for name, module in flux_transformer.named_modules():
            if "transformer_blocks" in name:
                module.requires_grad_(True)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for model in models:
                    if isinstance(unwrap_model(model), type(unwrap_model(flux_transformer))):
                        model = unwrap_model(model)
                        model.save_pretrained(os.path.join(output_dir, "transformer"))
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            transformer_ = None

            if not accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()

                    if isinstance(unwrap_model(model), type(unwrap_model(flux_transformer))):
                        transformer_ = model  # noqa: F841
                    else:
                        raise ValueError(f"unexpected save model: {unwrap_model(model).__class__}")

            else:
                transformer_ = FluxTransformer2DModel.from_pretrained(input_dir, subfolder="transformer")  # noqa: F841

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        flux_transformer.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.per_gpu_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimization parameters
    optimizer = optimizer_class(
        flux_transformer.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Prepare dataset and dataloader.
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes
    num_batches = math.ceil(args.num_train_examples / args.global_batch_size)
    dataset_obj = ControlFluxDataset(args)
    train_dataset = dataset_obj.get_dataset()
    train_dataloader = dataset_obj.prepare_dataloader(train_dataset)

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(args.num_train_examples / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    # Prepare everything with our `accelerator`.
    flux_transformer, optimizer, lr_scheduler = accelerator.prepare(
        flux_transformer, optimizer, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(num_batches / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' ({args.num_train_examples}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.per_gpu_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {args.num_train_examples}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Create a pipeline for text encoding. We will move this pipeline to GPU/CPU as needed.
    text_encoding_pipeline = FluxControlPipeline.from_pretrained(
        args.pretrained_model_name_or_path, transformer=None, vae=None, torch_dtype=weight_dtype
    )
    if not args.offload:
        text_encoding_pipeline = text_encoding_pipeline.to("cuda")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    if accelerator.is_main_process and args.report_to == "wandb" and args.log_dataset_samples:
        log_data_samples_to_wandb(dataloader=train_dataloader, accelerator=accelerator)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices]
        return expand_tensor_to_dim(sigma, n_dim)

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        flux_transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(flux_transformer):
                # Convert images to latent space
                # vae encode
                if args.offload:
                    vae.to(accelerator.device)
                edited_pixel_values = batch["edited_pixel_values"].to(dtype=vae.dtype, device=accelerator.device, non_blocking=True)
                pixel_latents = encode_images(edited_pixel_values, vae, weight_dtype)
                conditioning_pixel_values = batch["source_pixel_values"].to(
                    dtype=vae.dtype, device=accelerator.device, non_blocking=True
                )
                control_latents = encode_images(conditioning_pixel_values, vae, weight_dtype)
                if args.offload:
                    # offload vae to CPU.
                    vae.cpu()

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                bsz = pixel_latents.shape[0]
                noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)
                if args.simplified_flow:
                    sigmas = torch.rand(bsz, device=accelerator.device, dtype=pixel_latents.dtype)
                    timesteps = (sigmas * noise_scheduler_copy.config.num_train_timesteps).long().to(accelerator.device)
                    sigmas = expand_tensor_to_dim(sigmas, pixel_latents.ndim)
                else:
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)
                    sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)

                # Add noise according to flow matching.
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
                concatenated_noisy_model_input = torch.cat([noisy_model_input, control_latents], dim=1)

                # pack the latents.
                packed_noisy_model_input = FluxControlPipeline._pack_latents(
                    concatenated_noisy_model_input,
                    batch_size=bsz,
                    num_channels_latents=concatenated_noisy_model_input.shape[1],
                    height=concatenated_noisy_model_input.shape[2],
                    width=concatenated_noisy_model_input.shape[3],
                )

                # latent image ids for RoPE.
                latent_image_ids = FluxControlPipeline._prepare_latent_image_ids(
                    bsz,
                    concatenated_noisy_model_input.shape[2] // 2,
                    concatenated_noisy_model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )

                # handle guidance
                if unwrap_model(flux_transformer).config.guidance_embeds:
                    guidance_vec = torch.full(
                        (bsz,),
                        args.guidance_scale,
                        device=noisy_model_input.device,
                        dtype=weight_dtype,
                    )
                else:
                    guidance_vec = None

                # text encoding.
                captions = batch["captions"]
                if args.offload:
                    text_encoding_pipeline = text_encoding_pipeline.to("cuda")
                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
                        captions, prompt_2=None
                    )
                # this could be optimized by not having to do any text encoding and just
                # doing zeros on specified shapes for `prompt_embeds` and `pooled_prompt_embeds`
                if args.proportion_empty_prompts and random.random() < args.proportion_empty_prompts:
                    prompt_embeds.zero_()
                    pooled_prompt_embeds.zero_()
                if args.offload:
                    text_encoding_pipeline = text_encoding_pipeline.to("cpu")

                # Predict.
                model_pred = flux_transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                model_pred = FluxControlPipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[2] * vae_scale_factor,
                    width=noisy_model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow-matching loss
                target = noise - pixel_latents
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = flux_transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            flux_transformer=flux_transformer,
                            args=args,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            step=global_step,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        flux_transformer = unwrap_model(flux_transformer)
        if args.upcast_before_saving:
            flux_transformer.to(torch.float32)
        flux_transformer.save_pretrained(args.output_dir)

        del flux_transformer
        del text_encoding_pipeline
        del vae
        free_memory()

        # Run a final round of validation.
        image_logs = None
        if args.validation_prompt is not None:
            image_logs = log_validation(
                flux_transformer=None,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                step=global_step,
                is_final_validation=True,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*", "checkpoint-*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
