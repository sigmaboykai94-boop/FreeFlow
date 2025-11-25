# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
import argparse

from models import DiT_models
from utils import create_student

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    learn_sigma = True if args.image_size == 256 and "REPA" not in args.ckpt else False
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "")
    latent_size = args.image_size // 8

    # Load model:
    model = create_student(args, device)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    class_labels = [604, 928, 387, 974, 933, 979, 417, 279, 113, 812, 980, 475]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)
    # Sample images:
    t_steps = torch.linspace(1., 0., 2, device=device)

    # Sample images:
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
    os.makedirs("visuals", exist_ok=True)

    # Save and display images:
    save_image(samples, f"visuals/sample_{ckpt_string_name}_cfg_{args.cfg_scale}_seed_{args.seed}.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--num-sampling-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Model checkpoint path.")
    args = parser.parse_args()
    main(args)
