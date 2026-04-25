from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import open3d as o3d
import torch
import trimesh as tr
from PIL import Image
from scipy.spatial import cKDTree

from robot_lerf.clip_process import list_scene_images
from robot_lerf.siglip2_encoder import DEFAULT_SIGLIP2_MODEL, SigLIP2Encoder


def _slugify_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def _infer_scene_aliases(scene_name: str) -> list[str]:
    aliases = [scene_name]
    patterns = [
        r"^(.*?)(_robotinit(?:_v\d+)?)$",
        r"^(.*?)(_hloc(?:_v\d+)?)$",
        r"^(.*?)(_sfm(?:_v\d+)?)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, scene_name)
        if match:
            base_name = match.group(1)
            if base_name and base_name not in aliases:
                aliases.append(base_name)
    return aliases


def _frame_lookup_keys(file_path: str) -> list[str]:
    normalized = file_path.replace("\\", "/").lstrip("./")
    basename = Path(normalized).name
    keys = [normalized]
    if basename not in keys:
        keys.append(basename)
    images_prefixed = f"images/{basename}"
    if images_prefixed not in keys:
        keys.append(images_prefixed)
    return keys


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    min_val = float(scores.min())
    max_val = float(scores.max())
    if max_val - min_val < 1e-6:
        return np.ones_like(scores, dtype=np.float32)
    return (scores - min_val) / (max_val - min_val)


def _mask_top_fraction(scores: np.ndarray, quantile: float, min_points: int) -> np.ndarray:
    if scores.size == 0:
        return np.zeros(0, dtype=bool)
    threshold = float(np.quantile(scores, quantile))
    mask = scores >= threshold
    if int(mask.sum()) >= min_points:
        return mask
    top_k = min(min_points, scores.size)
    indices = np.argpartition(scores, -top_k)[-top_k:]
    dense_mask = np.zeros_like(scores, dtype=bool)
    dense_mask[indices] = True
    return dense_mask


def _cap_mask_size(scores: np.ndarray, mask: np.ndarray, max_points: int) -> np.ndarray:
    if int(mask.sum()) <= max_points:
        return mask
    selected = np.flatnonzero(mask)
    chosen = selected[np.argpartition(scores[selected], -max_points)[-max_points:]]
    capped_mask = np.zeros_like(mask, dtype=bool)
    capped_mask[chosen] = True
    return capped_mask


def _weighted_dbscan_mask(
    points: np.ndarray,
    scores: np.ndarray,
    eps: float,
    min_points: int,
) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    if points.shape[0] < min_points:
        return np.ones(points.shape[0], dtype=bool)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    labels = np.asarray(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False),
        dtype=np.int64,
    )
    valid = labels >= 0
    if not np.any(valid):
        best_idx = int(np.argmax(scores))
        distances = np.linalg.norm(points - points[best_idx], axis=1)
        radius = max(eps * 1.5, float(np.quantile(distances, 0.15)))
        fallback_mask = distances <= radius
        if int(fallback_mask.sum()) >= max(16, min_points // 2):
            return fallback_mask
        return np.ones(points.shape[0], dtype=bool)

    best_label = -1
    best_value = -np.inf
    for label in np.unique(labels[valid]):
        label_mask = labels == label
        score_sum = float(scores[label_mask].sum())
        score_mean = float(scores[label_mask].mean())
        cluster_size = int(label_mask.sum())
        value = score_sum + 0.25 * score_mean * cluster_size
        if value > best_value:
            best_value = value
            best_label = int(label)
    return labels == best_label


def _refine_mask_to_cluster(
    points: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray,
    eps: float,
    min_points: int,
) -> np.ndarray:
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return mask
    keep_local = _weighted_dbscan_mask(points[selected], scores[selected], eps=eps, min_points=min_points)
    refined = np.zeros_like(mask, dtype=bool)
    refined[selected[keep_local]] = True
    return refined


def _normalize_rows(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.size == 0:
        return features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    return features / norms


def _seeded_descriptor_mask(
    points: np.ndarray,
    descriptors: np.ndarray,
    seed_index: int,
    spatial_radius: float,
    min_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float32)
    descriptors = _normalize_rows(descriptors)
    seed_index = int(np.clip(seed_index, 0, points.shape[0] - 1))
    seed_feature = descriptors[seed_index]
    descriptor_scores = np.clip((descriptors @ seed_feature + 1.0) * 0.5, 0.0, 1.0)

    similarity_threshold = float(
        np.clip(np.quantile(descriptor_scores, 0.9), 0.35, 0.85)
    )
    tree = cKDTree(points)
    visited = np.zeros(points.shape[0], dtype=bool)
    keep = np.zeros(points.shape[0], dtype=bool)
    frontier = [seed_index]

    while frontier:
        current = frontier.pop()
        if visited[current]:
            continue
        visited[current] = True
        if descriptor_scores[current] < similarity_threshold:
            continue
        keep[current] = True
        neighbors = tree.query_ball_point(points[current], r=spatial_radius)
        for neighbor in neighbors:
            if not visited[neighbor] and descriptor_scores[neighbor] >= similarity_threshold:
                frontier.append(int(neighbor))

    if int(keep.sum()) < min_points:
        keep = _mask_top_fraction(descriptor_scores, quantile=0.94, min_points=min_points)
        keep = _refine_mask_to_cluster(
            points,
            descriptor_scores,
            keep,
            eps=spatial_radius,
            min_points=max(12, min_points // 2),
        )
    return keep, descriptor_scores.astype(np.float32)


@dataclass
class SemanticTileRecord:
    points_model: np.ndarray
    colors: np.ndarray
    object_scores: np.ndarray
    composite_scores: np.ndarray
    place_scores: np.ndarray | None
    dino_features: np.ndarray | None = None
    dino_valid: np.ndarray | None = None


class SigLIPSceneQueryEngine:
    def __init__(
        self,
        scene_dir: Path,
        applied_transform: np.ndarray,
        output_root: Path | None = None,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self.scene_dir = self._resolve_scene_dir(Path(scene_dir).resolve())
        if not self.scene_dir.is_dir():
            raise FileNotFoundError(f"Scene directory not found: {self.scene_dir}")

        self.scene_name = self.scene_dir.name
        self._scene_name_candidates = _infer_scene_aliases(self.scene_name)
        self.output_root = (
            Path(output_root).resolve()
            if output_root is not None
            else self.scene_dir.parent.parent / "outputs"
        )
        self.model_name = model_name or os.environ.get(
            "ROBOT_LERF_SIGLIP2_MODEL",
            DEFAULT_SIGLIP2_MODEL,
        )
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.applied_transform = np.asarray(applied_transform, dtype=np.float64)
        self.world_to_model = np.linalg.inv(self.applied_transform)
        self._positives: list[str] = []
        self._reference_points_model: np.ndarray | None = None
        self._reference_colors: np.ndarray | None = None
        self._reference_assignments: np.ndarray | None = None
        self._reference_bind_radius = float(os.environ.get("ROBOT_LERF_SIGLIP_BIND_RADIUS", "0.01"))
        self._dino_points_model: np.ndarray | None = None
        self._dino_features: np.ndarray | None = None
        self._reference_dino_features: np.ndarray | None = None
        self._reference_has_dino: np.ndarray | None = None
        self._dino_bind_radius = float(os.environ.get("ROBOT_LERF_DINO_BIND_RADIUS", "0.02"))
        self._encoder = SigLIP2Encoder(model_name=self.model_name, device=self.device)
        self._load_scene_aliases()
        self._semantic_scene_dir = self._discover_semantic_scene_dir()
        self._semantic_scene_name = self._semantic_scene_dir.name
        self._load_scene_metadata()
        self._load_embedding_tiles()
        self._load_dino_features()

    @property
    def has_embeddings(self) -> bool:
        return self._embedding_matrix is not None and self._embedding_matrix.shape[0] > 0

    def set_positives(self, positives: Sequence[str]) -> None:
        cleaned = [positive.strip() for positive in positives if positive and positive.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty SigLIP query term is required.")
        self._positives = cleaned

    def get_lerf_pointcloud(self):
        if not self._positives:
            raise ValueError("Query positives have not been set.")
        if not self.has_embeddings:
            raise RuntimeError("No SigLIP scene embeddings were loaded for this scene.")

        text_features = self._encoder.encode_text(self._positives).detach().cpu().numpy()
        similarities = self._embedding_matrix @ text_features.T
        similarities = (similarities + 1.0) * 0.5
        normalized = np.stack(
            [_normalize_scores(similarities[:, idx]) for idx in range(similarities.shape[1])],
            axis=1,
        )

        record = self._build_semantic_record(normalized)
        return self._to_pointcloud_outputs(record)

    def set_reference_pointcloud(
        self,
        world_points: np.ndarray,
        colors: np.ndarray | None = None,
    ) -> None:
        world_points = np.asarray(world_points, dtype=np.float64)
        if world_points.size == 0:
            return
        points_model = tr.transformations.transform_points(world_points, self.world_to_model)
        self._reference_points_model = np.asarray(points_model, dtype=np.float32)
        if colors is None or len(colors) != len(world_points):
            self._reference_colors = np.zeros_like(self._reference_points_model, dtype=np.float32)
        else:
            self._reference_colors = np.asarray(colors, dtype=np.float32)
        self._bind_projected_points_to_reference()
        self._bind_dino_points_to_reference()

    def _bind_projected_points_to_reference(self) -> None:
        if self._reference_points_model is None or self._points_model.size == 0:
            self._reference_assignments = None
            return
        tree = cKDTree(self._reference_points_model)
        distances, indices = tree.query(self._points_model, k=1)
        valid = np.isfinite(distances) & (distances <= self._reference_bind_radius)
        assignments = np.full(self._points_model.shape[0], -1, dtype=np.int64)
        assignments[valid] = indices[valid].astype(np.int64)
        self._reference_assignments = assignments

    def _build_semantic_record(self, normalized: np.ndarray) -> SemanticTileRecord:
        if self._reference_assignments is None or self._reference_points_model is None:
            object_scores = normalized[:, 0]
            if normalized.shape[1] >= 2:
                composite_scores = object_scores * normalized[:, 1]
            else:
                composite_scores = object_scores.copy()
            place_scores = normalized[:, 2] if normalized.shape[1] >= 3 else None
            return SemanticTileRecord(
                points_model=self._points_model,
                colors=self._colors,
                object_scores=object_scores,
                composite_scores=composite_scores,
                place_scores=place_scores,
                dino_features=None,
                dino_valid=None,
            )

        valid = self._reference_assignments >= 0
        if not np.any(valid):
            return SemanticTileRecord(
                points_model=self._points_model,
                colors=self._colors,
                object_scores=normalized[:, 0],
                composite_scores=normalized[:, 0] * normalized[:, 1] if normalized.shape[1] >= 2 else normalized[:, 0].copy(),
                place_scores=normalized[:, 2] if normalized.shape[1] >= 3 else None,
                dino_features=None,
                dino_valid=None,
            )

        num_ref = self._reference_points_model.shape[0]
        aggregated_terms = []
        for term_index in range(normalized.shape[1]):
            reduced = np.zeros(num_ref, dtype=np.float32)
            np.maximum.at(
                reduced,
                self._reference_assignments[valid],
                normalized[valid, term_index].astype(np.float32),
            )
            aggregated_terms.append(reduced)

        observed = np.bincount(self._reference_assignments[valid], minlength=num_ref) > 0
        object_scores = aggregated_terms[0][observed]
        if len(aggregated_terms) >= 2:
            composite_scores = object_scores * aggregated_terms[1][observed]
        else:
            composite_scores = object_scores.copy()
        place_scores = aggregated_terms[2][observed] if len(aggregated_terms) >= 3 else None

        return SemanticTileRecord(
            points_model=self._reference_points_model[observed],
            colors=self._reference_colors[observed],
            object_scores=object_scores,
            composite_scores=composite_scores,
            place_scores=place_scores,
            dino_features=self._reference_dino_features[observed] if self._reference_dino_features is not None else None,
            dino_valid=self._reference_has_dino[observed] if self._reference_has_dino is not None else None,
        )

    def _to_pointcloud_outputs(self, record: SemanticTileRecord):
        object_gate_scores = record.object_scores.copy()
        semantic_gate_scores = record.composite_scores.copy()
        dino_gate_mask = None

        if record.dino_features is not None and record.dino_valid is not None and np.any(record.dino_valid):
            dino_indices = np.flatnonzero(record.dino_valid)
            seed_local = int(np.argmax(record.object_scores[dino_indices]))
            dino_keep_local, dino_descriptor_scores = _seeded_descriptor_mask(
                record.points_model[dino_indices],
                record.dino_features[dino_indices],
                seed_index=seed_local,
                spatial_radius=float(os.environ.get("ROBOT_LERF_DINO_SPATIAL_RADIUS", "0.03")),
                min_points=int(os.environ.get("ROBOT_LERF_DINO_MIN_POINTS", "96")),
            )
            dino_scores = np.zeros_like(record.object_scores, dtype=np.float32)
            dino_scores[dino_indices] = dino_descriptor_scores
            dino_gate_mask = np.zeros_like(record.object_scores, dtype=bool)
            dino_gate_mask[dino_indices[dino_keep_local]] = True
            object_gate_scores = object_gate_scores * np.clip(dino_scores, 0.0, 1.0)
            semantic_gate_scores = semantic_gate_scores * np.clip(dino_scores, 0.0, 1.0)

        object_mask = _mask_top_fraction(object_gate_scores, quantile=0.9, min_points=180)
        object_mask = _cap_mask_size(object_gate_scores, object_mask, max_points=2500)
        object_mask = _refine_mask_to_cluster(
            record.points_model,
            object_gate_scores,
            object_mask,
            eps=0.025,
            min_points=24,
        )

        if dino_gate_mask is not None:
            dino_object_mask = object_mask & dino_gate_mask
            if int(dino_object_mask.sum()) >= 48:
                object_mask = dino_object_mask

        semantic_mask = _mask_top_fraction(semantic_gate_scores, quantile=0.97, min_points=96)
        semantic_mask = _cap_mask_size(semantic_gate_scores, semantic_mask, max_points=1200)
        semantic_mask = _refine_mask_to_cluster(
            record.points_model,
            semantic_gate_scores,
            semantic_mask,
            eps=0.02,
            min_points=18,
        )

        if dino_gate_mask is not None:
            dino_semantic_mask = semantic_mask & dino_gate_mask
            if int(dino_semantic_mask.sum()) >= 24:
                semantic_mask = dino_semantic_mask
            elif int(object_mask.sum()) >= 24:
                semantic_mask = object_mask.copy()

        semantic_points = record.points_model[semantic_mask]
        semantic_colors = record.colors[semantic_mask]
        semantic_scores = semantic_gate_scores[semantic_mask]

        object_points = record.points_model[object_mask]
        object_colors = record.colors[object_mask]

        if semantic_points.shape[0] == 0 and object_points.shape[0] > 0:
            semantic_points = object_points
            semantic_colors = object_colors
            semantic_scores = object_gate_scores[object_mask]
            semantic_mask = object_mask.copy()

        semantic_pcd = o3d.geometry.PointCloud()
        semantic_pcd.points = o3d.utility.Vector3dVector(semantic_points.astype(np.float64))
        semantic_pcd.colors = o3d.utility.Vector3dVector(semantic_colors.astype(np.float64))

        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(object_points.astype(np.float64))
        object_pcd.colors = o3d.utility.Vector3dVector(object_colors.astype(np.float64))

        if len(semantic_points) > 32:
            semantic_pcd, keep = semantic_pcd.remove_statistical_outlier(
                nb_neighbors=20,
                std_ratio=0.5,
            )
            keep = np.asarray(keep, dtype=np.int64)
            semantic_scores = semantic_scores[keep]
        semantic_scores = _normalize_scores(semantic_scores).reshape(-1, 1)

        if len(object_points) > 32:
            object_pcd, keep = object_pcd.remove_statistical_outlier(
                nb_neighbors=20,
                std_ratio=0.75,
            )

        if np.any(object_mask):
            pick_scores = object_gate_scores.copy()
            pick_scores[~object_mask] = -np.inf
            pick_idx = int(np.argmax(pick_scores))
        else:
            pick_idx = int(np.argmax(object_gate_scores))
        pick_point_model = torch.from_numpy(record.points_model[pick_idx].astype(np.float32))

        place_point_model = None
        if record.place_scores is not None:
            if np.any(semantic_mask):
                place_scores = record.place_scores.copy()
                place_scores[~semantic_mask] = -np.inf
                place_idx = int(np.argmax(place_scores))
            else:
                place_idx = int(np.argmax(record.place_scores))
            place_point_model = torch.from_numpy(record.points_model[place_idx].astype(np.float32))

        return semantic_pcd, semantic_scores, object_pcd, pick_point_model, place_point_model

    def _load_scene_metadata(self) -> None:
        transforms_path = self._semantic_scene_dir / "transforms.json"
        if not transforms_path.is_file():
            raise FileNotFoundError(f"Missing transforms.json in {self._semantic_scene_dir}")
        transforms = json.loads(transforms_path.read_text())
        self._fx = float(transforms["fl_x"])
        self._fy = float(transforms["fl_y"])
        self._cx = float(transforms["cx"])
        self._cy = float(transforms["cy"])
        self._height = int(transforms["h"])
        self._width = int(transforms["w"])
        frames = transforms["frames"]
        self._frame_by_name = {}
        for frame in frames:
            for key in _frame_lookup_keys(frame["file_path"]):
                self._frame_by_name[key] = frame
        self._image_paths = list_scene_images(self._semantic_scene_dir)
        if not self._image_paths:
            raise RuntimeError(
                f"No images found for semantic scene {self._semantic_scene_dir}. "
                "Expected image files at the scene root or under an images/ subdirectory."
            )

    def _load_embedding_tiles(self) -> None:
        clip_dir = self._find_clip_dir()
        level_files = sorted(clip_dir.glob("level_*.npy"))
        if not level_files:
            raise FileNotFoundError(f"No SigLIP level embeddings found in {clip_dir}")

        points_world: list[np.ndarray] = []
        colors: list[np.ndarray] = []
        embeddings: list[np.ndarray] = []

        for level_file in level_files:
            level_index = int(level_file.stem.split("_")[-1])
            info_path = clip_dir / f"level_{level_index}.info"
            if not info_path.is_file():
                raise FileNotFoundError(f"Missing level info file: {info_path}")
            level_info = json.loads(info_path.read_text())
            tile_ratio = float(level_info["tile_ratio"])
            stride_ratio = float(level_info["stride_ratio"])
            level_embeddings = np.load(level_file)
            if level_embeddings.shape[0] != len(self._image_paths):
                raise RuntimeError(
                    f"SigLIP embeddings in {clip_dir} were generated for {level_embeddings.shape[0]} images, "
                    f"but semantic scene {self._semantic_scene_dir} has {len(self._image_paths)} images."
                )

            center_x, center_y = self._compute_tile_centers(tile_ratio, stride_ratio)
            for image_idx, image_path in enumerate(self._image_paths):
                frame = self._lookup_frame_for_image(image_path)
                if frame is None:
                    continue
                depth_path = self._resolve_depth_path(frame)
                if not depth_path.is_file():
                    continue
                depth = np.load(depth_path)
                image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
                c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
                points_i, colors_i, embeds_i = self._records_for_level_image(
                    depth=depth,
                    image=image,
                    c2w=c2w,
                    center_x=center_x,
                    center_y=center_y,
                    level_embeddings=level_embeddings[image_idx],
                )
                if points_i.size == 0:
                    continue
                points_world.append(points_i)
                colors.append(colors_i)
                embeddings.append(embeds_i)

        if not embeddings:
            raise RuntimeError(
                "No valid SigLIP tiles could be projected for scene "
                f"{self.scene_name}. Semantic metadata came from {self._semantic_scene_dir} "
                f"and embeddings came from scene aliases {self._scene_name_candidates}."
            )

        points_world_np = np.concatenate(points_world, axis=0)
        points_model_h = tr.transformations.transform_points(points_world_np, self.world_to_model)
        self._points_model = np.asarray(points_model_h, dtype=np.float32)
        self._colors = np.concatenate(colors, axis=0).astype(np.float32)
        self._embedding_matrix = np.concatenate(embeddings, axis=0).astype(np.float32)

    def _load_dino_features(self) -> None:
        dino_file = self._find_dino_file()
        if dino_file is None:
            self._dino_points_model = None
            self._dino_features = None
            return

        dino_grids = np.load(dino_file)
        if dino_grids.ndim != 4:
            self._dino_points_model = None
            self._dino_features = None
            return

        points_world: list[np.ndarray] = []
        descriptors: list[np.ndarray] = []
        num_images = min(len(self._image_paths), dino_grids.shape[0])
        for image_idx in range(num_images):
            image_path = self._image_paths[image_idx]
            frame = self._lookup_frame_for_image(image_path)
            if frame is None:
                continue
            depth_path = self._resolve_depth_path(frame)
            if not depth_path.is_file():
                continue
            depth = np.load(depth_path)
            c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
            descriptor_grid = np.asarray(dino_grids[image_idx], dtype=np.float32)
            center_x, center_y = self._compute_descriptor_centers(
                descriptor_grid.shape[0],
                descriptor_grid.shape[1],
            )
            points_i, descriptors_i = self._records_for_descriptor_image(
                depth=depth,
                c2w=c2w,
                center_x=center_x,
                center_y=center_y,
                descriptor_grid=descriptor_grid,
            )
            if points_i.size == 0:
                continue
            points_world.append(points_i)
            descriptors.append(descriptors_i)

        if not descriptors:
            self._dino_points_model = None
            self._dino_features = None
            return

        dino_world_points = np.concatenate(points_world, axis=0)
        dino_model_points = tr.transformations.transform_points(dino_world_points, self.world_to_model)
        self._dino_points_model = np.asarray(dino_model_points, dtype=np.float32)
        self._dino_features = _normalize_rows(np.concatenate(descriptors, axis=0).astype(np.float32))

    def _bind_dino_points_to_reference(self) -> None:
        if (
            self._reference_points_model is None
            or self._dino_points_model is None
            or self._dino_features is None
            or self._dino_points_model.size == 0
        ):
            self._reference_dino_features = None
            self._reference_has_dino = None
            return

        tree = cKDTree(self._reference_points_model)
        distances, indices = tree.query(self._dino_points_model, k=1)
        valid = np.isfinite(distances) & (distances <= self._dino_bind_radius)
        if not np.any(valid):
            self._reference_dino_features = None
            self._reference_has_dino = None
            return

        num_ref = self._reference_points_model.shape[0]
        dino_dim = self._dino_features.shape[1]
        aggregated = np.zeros((num_ref, dino_dim), dtype=np.float32)
        counts = np.zeros(num_ref, dtype=np.int32)
        np.add.at(aggregated, indices[valid], self._dino_features[valid])
        np.add.at(counts, indices[valid], 1)

        observed = counts > 0
        aggregated[observed] /= counts[observed, None]
        aggregated[observed] = _normalize_rows(aggregated[observed])
        self._reference_dino_features = aggregated
        self._reference_has_dino = observed

    def _find_clip_dir(self) -> Path:
        checked_dirs = []
        for scene_name in self._scene_name_candidates:
            scene_output_dir = self.output_root / scene_name
            checked_dirs.append(scene_output_dir)
            preferred = scene_output_dir / f"clip_siglip2_{_slugify_model_name(self.model_name)}"
            if preferred.is_dir():
                return preferred
            candidates = sorted(scene_output_dir.glob("clip_siglip2_*"))
            if len(candidates) == 1:
                return candidates[0]
            if candidates:
                return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]
        checked_str = ", ".join(str(path) for path in checked_dirs)
        raise FileNotFoundError(
            "No SigLIP clip directory found. Checked: "
            f"{checked_str}. Expected a clip_siglip2_* directory."
        )

    def _find_dino_file(self) -> Path | None:
        for scene_name in self._scene_name_candidates:
            scene_output_dir = self.output_root / scene_name
            preferred = scene_output_dir / "dino_dino_vitb8.npy"
            if preferred.is_file():
                return preferred
            candidates = sorted(scene_output_dir.glob("dino_*.npy"))
            if len(candidates) == 1:
                return candidates[0]
            if candidates:
                return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]
        return None

    @staticmethod
    def _resolve_scene_dir(scene_dir: Path) -> Path:
        if (scene_dir / "transforms.json").is_file():
            return scene_dir
        if (scene_dir.parent / "transforms.json").is_file():
            return scene_dir.parent
        return scene_dir

    def _load_scene_aliases(self) -> None:
        alignment_candidates = []
        explicit_alignment = os.environ.get("ROBOT_LERF_ALIGNMENT_JSON")
        if explicit_alignment:
            alignment_candidates.append(Path(explicit_alignment).expanduser())
        alignment_candidates.append(self.scene_dir / "alignment.json")

        seen = set()
        for alignment_path in alignment_candidates:
            alignment_path = alignment_path.resolve()
            if alignment_path in seen or not alignment_path.is_file():
                continue
            seen.add(alignment_path)
            try:
                alignment = json.loads(alignment_path.read_text())
            except Exception:
                continue
            for key in ("robot_scene", "output_scene", "sfm_scene"):
                scene_value = alignment.get(key)
                if not scene_value:
                    continue
                scene_name = Path(scene_value).name
                if scene_name and scene_name not in self._scene_name_candidates:
                    self._scene_name_candidates.append(scene_name)

    def _discover_semantic_scene_dir(self) -> Path:
        candidate_dirs: list[Path] = []
        candidate_dirs.append(self.scene_dir)
        for scene_name in self._scene_name_candidates:
            sibling = self.scene_dir.parent / scene_name
            if sibling not in candidate_dirs:
                candidate_dirs.append(sibling)

        for candidate in candidate_dirs:
            resolved = self._resolve_scene_dir(candidate)
            transforms_path = resolved / "transforms.json"
            if not transforms_path.is_file():
                continue
            images = list_scene_images(resolved)
            if not images:
                continue
            try:
                transforms = json.loads(transforms_path.read_text())
            except Exception:
                continue
            frames = transforms.get("frames", [])
            if not frames:
                continue
            has_depth = any("depth_file_path" in frame for frame in frames)
            if has_depth:
                return resolved

        return self.scene_dir

    def _lookup_frame_for_image(self, image_path: Path) -> dict | None:
        relative_candidates = []
        try:
            relative_path = image_path.relative_to(self._semantic_scene_dir).as_posix()
            relative_candidates.extend(_frame_lookup_keys(relative_path))
        except ValueError:
            pass
        relative_candidates.extend(_frame_lookup_keys(image_path.name))
        seen = set()
        for key in relative_candidates:
            if key in seen:
                continue
            seen.add(key)
            frame = self._frame_by_name.get(key)
            if frame is not None:
                return frame
        return None

    def _resolve_depth_path(self, frame: dict) -> Path:
        depth_file = frame.get("depth_file_path")
        if depth_file is None:
            return Path("__missing_depth__")
        depth_path = Path(depth_file)
        if depth_path.is_absolute():
            return depth_path
        return self._semantic_scene_dir / depth_path

    def _compute_tile_centers(self, tile_ratio: float, stride_ratio: float):
        kernel_size = int(self._height * tile_ratio)
        stride = int(kernel_size * stride_ratio)
        padding = kernel_size // 2
        center_x = (
            (kernel_size - 1) / 2
            - padding
            + stride
            * np.arange(
                np.floor((self._height + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
            )
        )
        center_y = (
            (kernel_size - 1) / 2
            - padding
            + stride
            * np.arange(
                np.floor((self._width + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
            )
        )
        center_x = np.concatenate([center_x, center_x[[-1]]]).astype(np.int32)
        center_y = np.concatenate([center_y, center_y[[-1]]]).astype(np.int32)
        return center_x, center_y

    def _compute_descriptor_centers(self, grid_h: int, grid_w: int):
        center_x = ((np.arange(grid_h, dtype=np.float64) + 0.5) * self._height / grid_h).astype(np.int32)
        center_y = ((np.arange(grid_w, dtype=np.float64) + 0.5) * self._width / grid_w).astype(np.int32)
        return center_x, center_y

    def _records_for_level_image(
        self,
        depth: np.ndarray,
        image: np.ndarray,
        c2w: np.ndarray,
        center_x: np.ndarray,
        center_y: np.ndarray,
        level_embeddings: np.ndarray,
    ):
        grid_x, grid_y = np.meshgrid(center_x, center_y, indexing="ij")
        px = np.clip(grid_y.reshape(-1), 0, self._width - 1)
        py = np.clip(grid_x.reshape(-1), 0, self._height - 1)

        z = depth[py, px].reshape(-1)
        valid = np.isfinite(z) & (z > 0)
        if not np.any(valid):
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, level_embeddings.shape[-1]), dtype=np.float32),
            )

        px = px[valid].astype(np.float64)
        py = py[valid].astype(np.float64)
        z = z[valid].astype(np.float64)

        x_cam = (px - self._cx) / self._fx * z
        y_cam = (py - self._cy) / self._fy * z
        cam_points = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=1)
        world_points = (c2w @ cam_points.T).T[:, :3]

        flat_embeddings = level_embeddings.reshape(-1, level_embeddings.shape[-1])[valid]
        flat_colors = image[py.astype(np.int32), px.astype(np.int32)]
        return (
            world_points.astype(np.float32),
            flat_colors.astype(np.float32),
            flat_embeddings.astype(np.float32),
        )

    def _records_for_descriptor_image(
        self,
        depth: np.ndarray,
        c2w: np.ndarray,
        center_x: np.ndarray,
        center_y: np.ndarray,
        descriptor_grid: np.ndarray,
    ):
        grid_x, grid_y = np.meshgrid(center_x, center_y, indexing="ij")
        px = np.clip(grid_y.reshape(-1), 0, self._width - 1)
        py = np.clip(grid_x.reshape(-1), 0, self._height - 1)

        z = depth[py, px].reshape(-1)
        valid = np.isfinite(z) & (z > 0)
        if not np.any(valid):
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, descriptor_grid.shape[-1]), dtype=np.float32),
            )

        px = px[valid].astype(np.float64)
        py = py[valid].astype(np.float64)
        z = z[valid].astype(np.float64)

        x_cam = (px - self._cx) / self._fx * z
        y_cam = (py - self._cy) / self._fy * z
        cam_points = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=1)
        world_points = (c2w @ cam_points.T).T[:, :3]

        descriptors = descriptor_grid.reshape(-1, descriptor_grid.shape[-1])[valid]
        return world_points.astype(np.float32), descriptors.astype(np.float32)
