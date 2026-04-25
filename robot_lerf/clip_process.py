import json
import multiprocessing as mp
import os
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from robot_lerf.siglip2_encoder import DEFAULT_SIGLIP2_MODEL, SigLIP2Encoder


@dataclass
class PyramidEmbeddingConfig:
    patch_tile_size_range: Tuple[float, float] = (0.05, 0.5)
    patch_tile_size_res: int = 7
    patch_stride_scaler: float = 0.5

    @classmethod
    def from_env(cls) -> "PyramidEmbeddingConfig":
        tile_min = float(os.environ.get("ROBOT_LERF_PATCH_TILE_MIN", "0.05"))
        tile_max = float(os.environ.get("ROBOT_LERF_PATCH_TILE_MAX", "0.5"))
        tile_res = int(os.environ.get("ROBOT_LERF_PATCH_TILE_RES", "7"))
        stride_scaler = float(os.environ.get("ROBOT_LERF_PATCH_STRIDE_SCALER", "0.5"))
        return cls(
            patch_tile_size_range=(tile_min, tile_max),
            patch_tile_size_res=tile_res,
            patch_stride_scaler=stride_scaler,
        )


class SigLIP2PyramidEmbedder:
    def __init__(
        self,
        device: str,
        model_name: str = DEFAULT_SIGLIP2_MODEL,
        pyramid_config: Optional[PyramidEmbeddingConfig] = None,
        img_shape: Sequence[int] = (896, 1600),
    ) -> None:
        self.device = device
        self.model_name = model_name
        self.pyramid_config = pyramid_config or PyramidEmbeddingConfig.from_env()
        self.img_shape = [int(img_shape[0]), int(img_shape[1])]
        self.tile_sizes = torch.linspace(
            *self.pyramid_config.patch_tile_size_range,
            self.pyramid_config.patch_tile_size_res,
        )
        self.strider_scaler_list = [
            self._stride_scaler(tile_ratio.item(), self.pyramid_config.patch_stride_scaler)
            for tile_ratio in self.tile_sizes
        ]
        self.model = SigLIP2Encoder(model_name=self.model_name, device=self.device)

    def clip_model_name(self) -> str:
        return f"siglip2_{self.model_name.replace('/', '_').replace('-', '_')}"

    def get_level_json(self, level_index: int) -> str:
        return json.dumps(
            {
                "tile_ratio": self.tile_sizes[level_index].item(),
                "stride_ratio": self.strider_scaler_list[level_index],
                "image_shape": self.img_shape,
                "model_name": self.clip_model_name(),
            }
        )

    def get_pyramid_json(self) -> str:
        return json.dumps(
            {
                "tile_size_range": self.pyramid_config.patch_tile_size_range,
                "tile_size_res": self.pyramid_config.patch_tile_size_res,
                "stride_scaler": self.pyramid_config.patch_stride_scaler,
                "image_shape": self.img_shape,
                "model_name": self.clip_model_name(),
            }
        )

    def get_level_jsons(self) -> List[str]:
        return [self.get_level_json(i) for i in range(self.pyramid_config.patch_tile_size_res)]

    def process_image(self, img: np.ndarray) -> List[np.ndarray]:
        img_tensor = torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1).float() / 255.0
        embeddings = self._get_all_scales(img_tensor)
        return [embed.numpy() for embed in embeddings]

    def _get_level_params(self, level_index: int) -> Tuple[int, int, int]:
        stride_scaler = self.strider_scaler_list[level_index]
        kernel_size = int(self.img_shape[0] * self.tile_sizes[level_index])
        stride = int(kernel_size * stride_scaler)
        padding = kernel_size // 2
        return kernel_size, stride, padding

    @staticmethod
    def _stride_scaler(tile_ratio: float, stride_scaler: float) -> float:
        return float(np.interp(tile_ratio, [0.05, 0.15], [1.0, stride_scaler]))

    def _get_all_scales(self, img: torch.Tensor) -> List[torch.Tensor]:
        all_embeds: List[torch.Tensor] = []
        for level_index in range(self.pyramid_config.patch_tile_size_res):
            print(f"  embedding level {level_index + 1}/{self.pyramid_config.patch_tile_size_res}")
            embeds = self._get_one_scale(img, level_index)
            all_embeds.append(embeds.cpu())
        return all_embeds

    def _get_one_scale(self, img: torch.Tensor, level_index: int) -> torch.Tensor:
        kernel_size, stride, padding = self._get_level_params(level_index)
        unfold_func = torch.nn.Unfold(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        center_x = (
            (kernel_size - 1) / 2
            - padding
            + stride
            * np.arange(
                np.floor((self.img_shape[0] + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
            )
        )
        center_y = (
            (kernel_size - 1) / 2
            - padding
            + stride
            * np.arange(
                np.floor((self.img_shape[1] + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
            )
        )
        aug_imgs = img.unsqueeze(0)
        tiles = unfold_func(aug_imgs).permute(2, 0, 1).view(-1, 3, kernel_size, kernel_size)
        clip_embeds = self.model.encode_image(tiles).detach().cpu()
        clip_embeds = clip_embeds.reshape((center_x.shape[0], center_y.shape[0], -1))
        clip_embeds = torch.concat((clip_embeds, clip_embeds[:, [-1], :]), dim=1)
        clip_embeds = torch.concat((clip_embeds, clip_embeds[[-1], :, :]), dim=0)
        return clip_embeds


class ClipProcess(mp.Process):
    def __init__(
        self,
        out_queue,
        device,
        model_name=None,
        pyramid_config: Optional[PyramidEmbeddingConfig] = None,
        img_shape: Sequence[int] = (896, 1600),
    ):
        super().__init__()
        self.out_queue = out_queue
        self.in_queue = mp.Queue(maxsize=0)
        self.device = device
        self.model_name = model_name or DEFAULT_SIGLIP2_MODEL
        self.pyramid_config = pyramid_config
        self.img_shape = img_shape
        self.daemon = True

    def run(self):
        embedder = SigLIP2PyramidEmbedder(
            device=self.device,
            model_name=self.model_name,
            pyramid_config=self.pyramid_config,
            img_shape=self.img_shape,
        )
        while True:
            try:
                img = self.in_queue.get(timeout=0.01)
            except queue.Empty:
                continue
            if img is None:
                print("CLIP DONE")
                break
            print("Processing clip")
            self.out_queue.put(embedder.process_image(img))

    def kill(self):
        self.in_queue.put(None)


def list_scene_images(scene_dir: Path) -> List[Path]:
    valid_suffixes = {".jpg", ".jpeg", ".png"}
    scene_dir = Path(scene_dir)
    search_roots = [scene_dir, scene_dir / "images"]
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        images = sorted(
            [
                path
                for path in search_root.iterdir()
                if path.is_file() and path.suffix.lower() in valid_suffixes
            ]
        )
        if images:
            return images
    return []


if __name__ == "__main__":
    mp.set_start_method("spawn")
    q = mp.Queue()
    clipproc = ClipProcess(q, "cuda:0")
    clipproc.start()
    embedder = SigLIP2PyramidEmbedder("cuda:0")
    print(embedder.get_pyramid_json())
    clipproc.join()
