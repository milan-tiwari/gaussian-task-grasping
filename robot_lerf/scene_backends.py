from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from nerfstudio.pipelines.base_pipeline import Pipeline

from robot_lerf.graspnet_baseline.load_ns_model import NerfstudioWrapper, RealsenseCamera
from robot_lerf.siglip_scene_query import SigLIPSceneQueryEngine


class SceneBackend(ABC):
    """Minimal interface shared by scene representations used by the grasp UI."""

    @property
    @abstractmethod
    def applied_transform(self) -> np.ndarray:
        """Return the transform from model coordinates into world coordinates."""

    @abstractmethod
    def create_pointcloud(self):
        """Create the world/global point clouds consumed by grasp generation."""

    @abstractmethod
    def get_lerf_pointcloud(self, camera, render_lerf: bool = True):
        """Return semantic point cloud outputs for the current query."""

    @abstractmethod
    def visercam_to_model(self, c2w: np.ndarray) -> np.ndarray:
        """Convert a viser/world camera pose into the backend's camera convention."""

    @abstractmethod
    def set_positives(self, positives: Sequence[str]) -> None:
        """Update the backend's text query positives."""

    @abstractmethod
    def close(self) -> None:
        """Release backend resources where possible."""

    @property
    @abstractmethod
    def semantic_queries_supported(self) -> bool:
        """Whether this backend supports text-conditioned 3D semantic querying."""


class NerfstudioSceneBackend(SceneBackend):
    """Adapter around the original LERF/Nerfstudio implementation."""

    def __init__(self, scene_path: str | None = None, pipeline: Pipeline | None = None):
        if scene_path is not None:
            self._wrapper = NerfstudioWrapper(scene_path=scene_path)
        elif pipeline is not None:
            self._wrapper = NerfstudioWrapper(pipeline=pipeline)
        else:
            raise ValueError("Must provide either scene_path or pipeline")

    @property
    def applied_transform(self) -> np.ndarray:
        return self._wrapper.applied_transform

    def create_pointcloud(self):
        return self._wrapper.create_pointcloud()

    def get_lerf_pointcloud(self, camera, render_lerf: bool = True):
        return self._wrapper.get_lerf_pointcloud(camera, render_lerf=render_lerf)

    def visercam_to_model(self, c2w: np.ndarray) -> np.ndarray:
        return self._wrapper.visercam_to_ns(c2w)

    def set_positives(self, positives: Sequence[str]) -> None:
        self._wrapper.pipeline.image_encoder.set_positives(list(positives))

    def close(self) -> None:
        if hasattr(self._wrapper, "pipeline"):
            del self._wrapper.pipeline

    @property
    def semantic_queries_supported(self) -> bool:
        return True


class SplatfactoSceneBackend(SceneBackend):
    """Geometry-only Gaussian Splatting adapter using Nerfstudio's Splatfacto."""

    def __init__(self, scene_path: str | None = None, pipeline: Pipeline | None = None):
        if scene_path is not None:
            self._wrapper = NerfstudioWrapper(scene_path=scene_path)
        elif pipeline is not None:
            self._wrapper = NerfstudioWrapper(pipeline=pipeline)
        else:
            raise ValueError("Must provide either scene_path or pipeline")
        self._query_init_error: str | None = None
        self._query_engine = self._build_query_engine()

    @property
    def applied_transform(self) -> np.ndarray:
        return self._wrapper.applied_transform

    def create_pointcloud(self):
        world_pointcloud, global_pointcloud, table_center = self._wrapper.create_pointcloud()
        if self._query_engine is not None:
            self._query_engine.set_reference_pointcloud(
                world_points=np.asarray(global_pointcloud.vertices),
                colors=np.asarray(global_pointcloud.colors),
            )
        return world_pointcloud, global_pointcloud, table_center

    def get_lerf_pointcloud(self, camera, render_lerf: bool = True):
        if self._query_engine is None:
            raise NotImplementedError(
                self._query_init_error
                or "Gaussian semantics require a SigLIP embedding directory under "
                "outputs/<scene>/clip_siglip2_... plus the transformers dependency."
            )
        return self._query_engine.get_lerf_pointcloud()

    def visercam_to_model(self, c2w: np.ndarray) -> np.ndarray:
        return self._wrapper.visercam_to_ns(c2w)

    def set_positives(self, positives: Sequence[str]) -> None:
        if self._query_engine is None:
            raise NotImplementedError(
                self._query_init_error
                or "Gaussian semantics require a SigLIP embedding directory under "
                "outputs/<scene>/clip_siglip2_... plus the transformers dependency."
            )
        self._query_engine.set_positives(positives)

    def close(self) -> None:
        if hasattr(self._wrapper, "pipeline"):
            del self._wrapper.pipeline

    @property
    def semantic_queries_supported(self) -> bool:
        return self._query_engine is not None

    def _build_query_engine(self) -> SigLIPSceneQueryEngine | None:
        try:
            image_filenames = self._wrapper.pipeline.datamanager.train_dataparser_outputs.image_filenames
        except Exception:
            self._query_init_error = "Gaussian semantic query setup could not read scene image filenames from the loaded pipeline."
            return None
        if not image_filenames:
            self._query_init_error = "Gaussian semantic query setup found no training image filenames in the loaded pipeline."
            return None
        scene_dir = Path(image_filenames[0]).resolve().parent
        try:
            return SigLIPSceneQueryEngine(
                scene_dir=scene_dir,
                applied_transform=self.applied_transform,
            )
        except Exception as exc:
            self._query_init_error = str(exc)
            return None


def create_scene_backend(
    backend_type: str,
    scene_path: str | None = None,
    pipeline: Pipeline | None = None,
) -> SceneBackend:
    """Factory for scene representations.

    Gaussian Splatting support will plug into this same interface next.
    """
    normalized = backend_type.lower()
    if normalized in {"lerf", "nerfstudio"}:
        return NerfstudioSceneBackend(scene_path=scene_path, pipeline=pipeline)
    if normalized in {"gaussian", "gaussian-splatting", "gaussian_splatting", "gs"}:
        return SplatfactoSceneBackend(scene_path=scene_path, pipeline=pipeline)
    raise ValueError(f"Unknown scene backend '{backend_type}'")


def training_method_name_for_backend(backend_type: str) -> str:
    normalized = backend_type.lower()
    if normalized in {"lerf", "nerfstudio"}:
        return "lerf-lite"
    if normalized in {"gaussian", "gaussian-splatting", "gaussian_splatting", "gs"}:
        return "splatfacto"
    raise ValueError(f"Unknown scene backend '{backend_type}'")


def backend_runtime_error(backend_type: str) -> str | None:
    normalized = backend_type.lower()
    if normalized in {"gaussian", "gaussian-splatting", "gaussian_splatting", "gs"} and not torch.cuda.is_available():
        return (
            "The installed Gaussian backend currently requires CUDA in this Nerfstudio version. "
            "This machine does not have CUDA available, so use the nerfstudio/LERF backend locally "
            "or run Gaussian Splatting on a CUDA machine."
        )
    return None


__all__ = [
    "SceneBackend",
    "NerfstudioSceneBackend",
    "SplatfactoSceneBackend",
    "RealsenseCamera",
    "backend_runtime_error",
    "create_scene_backend",
    "training_method_name_for_backend",
]
