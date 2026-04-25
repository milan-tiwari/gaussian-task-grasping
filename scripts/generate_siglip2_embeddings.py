import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot_lerf.clip_process import (
    DEFAULT_SIGLIP2_MODEL,
    PyramidEmbeddingConfig,
    SigLIP2PyramidEmbedder,
    list_scene_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SigLIP 2 multi-scale embeddings for a captured scene."
    )
    parser.add_argument("--scene-dir", required=True, help="Path to the scene image folder.")
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root folder where outputs/<scene>/clip_<model>/... will be written.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_SIGLIP2_MODEL,
        help="Hugging Face SigLIP 2 model id.",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--image-height", type=int, default=896, help="Expected image height.")
    parser.add_argument("--image-width", type=int, default=1600, help="Expected image width.")
    parser.add_argument("--tile-min", type=float, default=0.05, help="Minimum tile ratio.")
    parser.add_argument("--tile-max", type=float, default=0.5, help="Maximum tile ratio.")
    parser.add_argument("--tile-res", type=int, default=7, help="Number of pyramid levels.")
    parser.add_argument(
        "--stride-scaler",
        type=float,
        default=0.5,
        help="Stride scaling for larger tiles.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    scene_name = scene_dir.name
    pyramid_config = PyramidEmbeddingConfig(
        patch_tile_size_range=(args.tile_min, args.tile_max),
        patch_tile_size_res=args.tile_res,
        patch_stride_scaler=args.stride_scaler,
    )
    embedder = SigLIP2PyramidEmbedder(
        device=args.device,
        model_name=args.model_name,
        pyramid_config=pyramid_config,
        img_shape=(args.image_height, args.image_width),
    )

    output_root = Path(args.output_root).expanduser().resolve()
    scene_output_dir = output_root / scene_name
    clipstring = embedder.clip_model_name()
    clip_output_dir = scene_output_dir / f"clip_{clipstring}"
    clip_output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_scene_images(scene_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {scene_dir}")

    per_level_embeddings = [[] for _ in range(pyramid_config.patch_tile_size_res)]
    for image_index, image_path in enumerate(image_paths, start=1):
        print(f"[{image_index}/{len(image_paths)}] {image_path.name}")
        image = np.asarray(Image.open(image_path).convert("RGB"))
        if image.shape[0] != args.image_height or image.shape[1] != args.image_width:
            raise ValueError(
                f"Image {image_path.name} has shape {image.shape[:2]}, "
                f"expected {(args.image_height, args.image_width)}"
            )
        level_embeddings = embedder.process_image(image)
        for level_index, level_embedding in enumerate(level_embeddings):
            per_level_embeddings[level_index].append(level_embedding)

    level_jsons = embedder.get_level_jsons()
    for level_index, level_embeddings in enumerate(per_level_embeddings):
        np.save(clip_output_dir / f"level_{level_index}.npy", np.stack(level_embeddings))
        (clip_output_dir / f"level_{level_index}.info").write_text(level_jsons[level_index])

    (scene_output_dir / f"clip_{clipstring}.info").write_text(embedder.get_pyramid_json())
    summary = {
        "scene_dir": str(scene_dir),
        "output_dir": str(clip_output_dir),
        "num_images": len(image_paths),
        "model_name": args.model_name,
        "device": args.device,
        "image_shape": [args.image_height, args.image_width],
        "tile_size_range": list(pyramid_config.patch_tile_size_range),
        "tile_size_res": pyramid_config.patch_tile_size_res,
        "stride_scaler": pyramid_config.patch_stride_scaler,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
