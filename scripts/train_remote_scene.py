#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Nerfstudio scene model for LERF or Gaussian Splatting on a remote machine."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the scene folder. Prefer a repo-relative path like data/flowers_take7.",
    )
    parser.add_argument(
        "--backend",
        choices=("nerfstudio", "lerf", "lerf-lite", "lerf-big", "gaussian"),
        default="gaussian",
        help=(
            "Scene backend / method to train. "
            "'gaussian' maps to Nerfstudio's splatfacto. "
            "'lerf' maps to the default LERF method, while 'lerf-lite' and 'lerf-big' "
            "select the explicit LERF variants."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30000,
        help="Maximum number of training iterations.",
    )
    parser.add_argument(
        "--run-name",
        default="sol_run",
        help="Timestamp/run name written into outputs/<scene>/<method>/<run-name>/.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "mps"),
        default="cuda",
        help="Training device type.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output root for checkpoints and configs.",
    )
    parser.add_argument(
        "--vis",
        choices=("tensorboard", "wandb", "comet", "viewer", "viewer+tensorboard"),
        default="tensorboard",
        help="Nerfstudio logging backend. tensorboard is a good default for clusters.",
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Disable torch.compile by monkeypatching it to a no-op. Useful for debugging cluster issues.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=2000,
        help="Checkpoint save interval.",
    )
    parser.add_argument(
        "--keep-all-checkpoints",
        action="store_true",
        help="Keep every saved checkpoint instead of only the latest one.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1000,
        help="Evaluation image/all-images interval. Set to 0 to disable eval during training.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start this run from scratch even if checkpoints already exist in the same run directory.",
    )
    return parser.parse_args()


def normalize_repo_relative(path_str: str, cwd: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        return path
    try:
        return path.relative_to(cwd)
    except ValueError:
        raise ValueError(
            "Use a path inside the repo checkout so the saved config stays portable. "
            f"Got absolute path '{path}'."
        ) from None


def method_name_for_backend(backend: str) -> str:
    if backend == "nerfstudio":
        return "lerf-lite"
    if backend == "lerf":
        return "lerf"
    if backend == "lerf-lite":
        return "lerf-lite"
    if backend == "lerf-big":
        return "lerf-big"
    if backend == "gaussian":
        return "splatfacto"
    raise ValueError(f"Unknown backend '{backend}'")


def maybe_disable_compile() -> None:
    import torch

    def no_compile(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        return lambda fn: fn

    torch.compile = no_compile


def maybe_enable_resume(config) -> Path | None:
    checkpoint_dir = config.get_checkpoint_dir()
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob("step-*.ckpt"))
    if not checkpoints:
        return None
    config.load_dir = checkpoint_dir
    config.load_step = None
    return checkpoints[-1]


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    data_path = normalize_repo_relative(args.data, repo_root)
    abs_data_path = (repo_root / data_path).resolve()

    if not abs_data_path.exists():
        raise FileNotFoundError(f"Scene folder not found: {abs_data_path}")
    if not (abs_data_path / "transforms.json").exists():
        raise FileNotFoundError(f"Missing transforms.json in scene folder: {abs_data_path}")

    if args.disable_compile:
        maybe_disable_compile()

    import nerfstudio.configs.method_configs as method_configs

    method_name = method_name_for_backend(args.backend)
    config = method_configs.all_methods[method_name]
    config.pipeline.datamanager.data = data_path
    config.output_dir = Path(args.output_dir)
    config.max_num_iterations = args.steps
    config.steps_per_save = args.save_every
    config.timestamp = args.run_name
    config.machine.device_type = args.device
    config.machine.num_devices = 1
    config.viewer.quit_on_train_completion = True
    config.vis = args.vis
    config.save_only_latest_checkpoint = not args.keep_all_checkpoints

    if args.eval_every <= 0:
        config.steps_per_eval_batch = 0
        if hasattr(config, "steps_per_eval_image"):
            config.steps_per_eval_image = 0
        if hasattr(config, "steps_per_eval_all_images"):
            config.steps_per_eval_all_images = 0
    else:
        if hasattr(config, "steps_per_eval_image"):
            config.steps_per_eval_image = args.eval_every
        if hasattr(config, "steps_per_eval_all_images"):
            config.steps_per_eval_all_images = args.eval_every

    os.makedirs(config.get_base_dir(), exist_ok=True)
    resumed_ckpt = None
    if not args.no_resume:
        resumed_ckpt = maybe_enable_resume(config)
    config.save_config()

    print(f"Backend: {args.backend} -> {method_name}")
    print(f"Data: {data_path}")
    print(f"Device: {args.device}")
    print(f"Run dir: {config.get_base_dir()}")
    print(f"Checkpoint dir: {config.get_checkpoint_dir()}")
    if resumed_ckpt is not None:
        print(f"Resuming from: {resumed_ckpt}")
    else:
        print("Resuming from: scratch")
    sys.stdout.flush()

    trainer = config.setup(local_rank=0, world_size=1)
    trainer.setup()

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    print(f"TRAIN_OK {elapsed:.2f}s")
    print(f"CONFIG_PATH {config.get_base_dir() / 'config.yml'}")
    print(f"CHECKPOINT_DIR {config.get_base_dir() / 'nerfstudio_models'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
