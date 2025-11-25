# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist

from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse

from models import DiT_models
from utils import create_student, create_teacher, build_label_sampler

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def teacher_sample_fn(model, vae, z, y, steps, cfg=1.0, device='cuda', args=None):

    t_steps = torch.linspace(1., 0., steps + 1, device=device)
    n = z.shape[0]

    for t_curr, t_next in zip(t_steps[:-1], t_steps[1:]):
        t = 1 - t_curr if "REPA" not in args.ckpt else t_curr
        gw = cfg if (t >= args.guidance_start and t <= args.guidance_end) else 1.0
        dt = t_curr - t_next if "REPA" not in args.ckpt else t_next - t_curr

        d_curr = model(z, torch.full((n,), t, device=device), y)
        if d_curr.shape[1] == 8:
            d_curr = d_curr.chunk(2, dim=1)[0]
        if gw > 1.0:
            unguide_d_curr = model(
                z, torch.full((n,), t, device=device), torch.full((n,), 1000, device=device),
            )
            if d_curr.shape[1] == 8:
                d_curr = d_curr.chunk(2, dim=1)[0]

            d_curr = unguide_d_curr + gw * (d_curr - unguide_d_curr)
        
        d_next = model(z + dt * d_curr, torch.full((n,), 1 - t_next if "REPA" not in args.ckpt else t_next, device=device), y)
        if d_next.shape[1] == 8:
            d_next = d_next.chunk(2, dim=1)[0]
        if gw > 1.0:
            unguide_d_next = model(
                z + dt * d_curr, torch.full((n,), 1 - t_next if "REPA" not in args.ckpt else t_next, device=device), torch.full((n,), 1000, device=device)
            )
            if d_next.shape[1] == 8:
                d_next = d_next.chunk(2, dim=1)[0]

            d_next = unguide_d_next + gw * (d_next - unguide_d_next)
        
        z = z + dt / 2 * (d_curr + d_next)
    
    return z


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000


    learn_sigma = True if args.image_size == 256 and "REPA" not in args.ckpt else False
    latent_size = args.image_size // 8
    model = create_teacher(args, device) if args.use_teacher else create_student(args, device)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}-steps-{args.num_sampling_steps}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    if not args.use_teacher:
        assert args.num_sampling_steps == 1, "Only 1 step is supported for student model sampling."
    t_steps = torch.linspace(1., 0., args.num_sampling_steps + 1, device=device)

    label_sampler = build_label_sampler(
        args.label_sampling_mode,
        args.num_classes,
        args.num_fid_samples,
        total_samples,
        samples_needed_this_gpu,
        n,
        device,
        rank=rank,
        iterations=iterations,
        seed=args.global_seed,
    )

    for step_idx in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = label_sampler(step_idx)

        # Sample images:
        if args.use_teacher:
            z = teacher_sample_fn(model, vae, z, y, args.num_sampling_steps, args.cfg_scale, device, args)
        else:
            for t_curr, t_next in zip(t_steps[:-1], t_steps[1:]):
                # Sample the next timestep:
                t = 1 - t_curr if "REPA" not in args.ckpt else t_curr
                direction = 1 if "REPA" not in args.ckpt else -1

                dt = t_curr - t_next
                model_pred = model(
                    z,
                    torch.full((n,), t, device=device),
                    y,
                    dt=torch.full((n,), dt, device=device),
                    gw=torch.full((n,), args.cfg_scale, device=device),
                )
                if learn_sigma:
                    model_pred, _ = model_pred.chunk(2, dim=1)
                z = z + direction * dt * model_pred
    
        samples = vae.decode(z / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

        dist.barrier()  # to avoid timeout error

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--guidance-start", type=float, default=0.0)
    parser.add_argument("--guidance-end", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=1)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Model checkpoint path.")
    parser.add_argument("--use-teacher", action="store_true", default=False,
                        help="Use teacher model for sampling.")
    parser.add_argument("--teacher-ckpt", type=str, default="SiT-XL",
                        help="Teacher model name.")
    parser.add_argument("--label-sampling-mode", type=str, choices=["random", "equal"], default="random",
                        help="Label sampling mode.")
    args = parser.parse_args()
    main(args)
