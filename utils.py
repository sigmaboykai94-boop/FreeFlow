import os

import torch

from models import DiT_models
from download import find_model


def create_student(args, device):

    learn_sigma = True if args.image_size == 256 and "REPA" not in args.ckpt else False

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=learn_sigma,
        take_dt=True,
        take_gw=True,
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)

    model.eval()  # important!

    return model

def create_teacher(args, device):

    learn_sigma = True if args.image_size == 256 and "REPA" not in args.ckpt else False

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=learn_sigma,
        take_dt=not args.use_teacher,
        take_gw=not args.use_teacher,
    ).to(device)

    ckpt_path = args.teacher_ckpt
    state_dict = find_model(ckpt_path)

    # update incompatible keys
    state_dict['t_embedder.projs.0.weight'] = state_dict['t_embedder.mlp.0.weight']
    state_dict['t_embedder.projs.0.bias'] = state_dict['t_embedder.mlp.0.bias']
    state_dict['t_embedder.projs.2.weight'] = state_dict['t_embedder.mlp.2.weight']
    state_dict['t_embedder.projs.2.bias'] = state_dict['t_embedder.mlp.2.bias']

    for params_name in state_dict.keys():
        if 't_embedder.mlp' in params_name:
            del state_dict[params_name]
        if 'projectors' in params_name:
            del state_dict[params_name]
    
    model.load_state_dict(state_dict)
    model.eval()

    return model


def build_label_sampler(
    sampling_mode: str,
    num_classes: int,
    num_fid_samples: int,
    total_samples: int,
    samples_needed_this_device: int,
    batch_size: int,
    device: torch.device,
    rank: int,
    iterations: int,
    seed: int,
):
    """Create a callable that returns a batch of labels for the given step index."""

    if sampling_mode == "random":
        def random_sampler(_step_idx: int) -> torch.Tensor:
            return torch.randint(0, num_classes, (batch_size,), device=device)

        return random_sampler

    if sampling_mode != "equal":
        raise ValueError(f"Unknown label sampling mode: {sampling_mode}")

    if num_fid_samples % num_classes != 0:
        raise ValueError(
            f"Equal label sampling requires num_fid_samples ({num_fid_samples}) to be divisible by num_classes ({num_classes})."
        )

    labels_per_class = num_fid_samples // num_classes
    base_pool = torch.arange(num_classes, dtype=torch.long).repeat_interleave(labels_per_class)

    generator = torch.Generator()
    generator.manual_seed(seed)
    permutation = torch.randperm(base_pool.numel(), generator=generator)
    base_pool = base_pool[permutation]

    if total_samples > num_fid_samples:
        tail = torch.randint(0, num_classes, (total_samples - num_fid_samples,), generator=generator)
        global_pool = torch.cat([base_pool, tail], dim=0)
    else:
        global_pool = base_pool

    start = rank * samples_needed_this_device
    end = start + samples_needed_this_device
    device_pool = global_pool[start:end]
    device_pool = device_pool.view(iterations, batch_size)

    def equal_sampler(step_idx: int) -> torch.Tensor:
        labels = device_pool[step_idx]
        return labels.to(device)

    return equal_sampler