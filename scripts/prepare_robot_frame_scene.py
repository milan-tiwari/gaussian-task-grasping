#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a robot/world-frame scene folder by aligning an SfM/HLOC scene to the "
            "original robot-frame dataset and transferring the sparse point cloud."
        )
    )
    parser.add_argument(
        "--robot-scene",
        required=True,
        help="Original scene folder with robot/world-frame transforms.json (for example data/flowers_take7).",
    )
    parser.add_argument(
        "--sfm-scene",
        required=True,
        help="SfM/HLOC scene folder with transforms.json and sparse reconstruction (for example data/flowers_take7_hloc_v3).",
    )
    parser.add_argument(
        "--output-scene",
        required=True,
        help="Output folder to create. This becomes the retraining dataset in robot/world coordinates.",
    )
    parser.add_argument(
        "--match-mode",
        choices=("sequence", "basename"),
        default="sequence",
        help=(
            "How to match source/target frames. 'sequence' pairs frames after natural sorting, "
            "which fits img0.jpg -> frame_00001.jpg style datasets."
        ),
    )
    parser.add_argument(
        "--source-sort",
        choices=("natural", "lexicographic"),
        default="natural",
        help="Sorting mode for source frames when using sequence matching.",
    )
    parser.add_argument(
        "--target-sort",
        choices=("natural", "lexicographic"),
        default="natural",
        help=(
            "Sorting mode for target frames when using sequence matching. "
            "Use lexicographic when the SfM scene was created by ns-process-data from filenames like img0/img1/img10."
        ),
    )
    parser.add_argument(
        "--source-ply",
        default=None,
        help=(
            "Optional explicit source point cloud path. If omitted, the script prefers the source "
            "scene's ply_file_path and falls back to colmap/sparse/0/points3D.bin."
        ),
    )
    parser.add_argument(
        "--target-ply-name",
        default="sparse_pc.ply",
        help="Filename to write inside the output scene for the aligned sparse point cloud.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output scene directory if it already exists.",
    )
    return parser.parse_args()


def natural_key(text: str) -> List[object]:
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", text)]


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_homogeneous(matrix_like: Sequence[Sequence[float]]) -> np.ndarray:
    mat = np.asarray(matrix_like, dtype=np.float64)
    if mat.shape == (4, 4):
        return mat
    if mat.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :] = mat
        return out
    raise ValueError(f"Expected a 3x4 or 4x4 transform, got shape {mat.shape}")


def frame_basename(frame: dict) -> str:
    return Path(frame["file_path"]).name


def sorted_frames(frames: Iterable[dict], sort_mode: str) -> List[dict]:
    if sort_mode == "natural":
        return sorted(frames, key=lambda frame: natural_key(frame_basename(frame)))
    if sort_mode == "lexicographic":
        return sorted(frames, key=lambda frame: frame_basename(frame))
    raise ValueError(f"Unsupported sort mode: {sort_mode}")


def match_frames(
    source_frames: List[dict],
    target_frames: List[dict],
    mode: str,
    source_sort: str,
    target_sort: str,
) -> List[Tuple[dict, dict]]:
    if mode == "sequence":
        source_sorted = sorted_frames(source_frames, source_sort)
        target_sorted = sorted_frames(target_frames, target_sort)
        if len(source_sorted) != len(target_sorted):
            raise ValueError(
                "Sequence matching requires the same number of source and target frames. "
                f"Got {len(source_sorted)} source and {len(target_sorted)} target frames."
            )
        return list(zip(source_sorted, target_sorted))

    if mode == "basename":
        source_by_name = {frame_basename(frame): frame for frame in source_frames}
        target_by_name = {frame_basename(frame): frame for frame in target_frames}
        common_names = sorted(source_by_name.keys() & target_by_name.keys(), key=natural_key)
        if not common_names:
            raise ValueError("No overlapping frame names found for basename matching.")
        return [(source_by_name[name], target_by_name[name]) for name in common_names]

    raise ValueError(f"Unsupported match mode: {mode}")


def camera_center_from_frame(frame: dict) -> np.ndarray:
    pose = ensure_homogeneous(frame["transform_matrix"])
    return pose[:3, 3]


def rotation_from_frame(frame: dict) -> np.ndarray:
    pose = ensure_homogeneous(frame["transform_matrix"])
    return pose[:3, :3]


def umeyama_similarity(source_points: np.ndarray, target_points: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    if source_points.shape != target_points.shape:
        raise ValueError(
            f"Source and target point arrays must match. Got {source_points.shape} vs {target_points.shape}."
        )
    if source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 point arrays, got {source_points.shape}")
    if len(source_points) < 3:
        raise ValueError("Need at least three corresponding camera centers for alignment.")

    src_mean = source_points.mean(axis=0)
    tgt_mean = target_points.mean(axis=0)
    src_centered = source_points - src_mean
    tgt_centered = target_points - tgt_mean

    cov = (tgt_centered.T @ src_centered) / len(source_points)
    u, singular_vals, vt = np.linalg.svd(cov)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt

    src_var = np.mean(np.sum(src_centered * src_centered, axis=1))
    if src_var <= 0:
        raise ValueError("Degenerate source camera centers; cannot estimate similarity transform.")
    scale = np.trace(np.diag(singular_vals) @ correction) / src_var
    translation = tgt_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def apply_similarity(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (scale * (rotation @ points.T)).T + translation


def read_next_bytes(fid, num_bytes: int, format_char_sequence: str, endian_character: str = "<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3d_binary(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    points_xyz: List[Tuple[float, float, float]] = []
    points_rgb: List[Tuple[int, int, int]] = []
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            elems = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = tuple(float(v) for v in elems[1:4])
            rgb = tuple(int(v) for v in elems[4:7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            fid.read(8 * track_length)
            points_xyz.append(xyz)
            points_rgb.append(rgb)
    return np.asarray(points_xyz, dtype=np.float64), np.asarray(points_rgb, dtype=np.uint8)


def read_ascii_ply(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        header: List[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading PLY header: {path}")
            line = line.rstrip("\n")
            header.append(line)
            if line == "end_header":
                break
        if not header or header[0] != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        if "format ascii 1.0" not in header:
            raise ValueError(f"Only ascii PLY is supported by this helper: {path}")

        vertex_count = None
        properties: List[str] = []
        in_vertex_block = False
        for line in header:
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
                in_vertex_block = True
                continue
            if line.startswith("element ") and not line.startswith("element vertex "):
                in_vertex_block = False
            if in_vertex_block and line.startswith("property "):
                properties.append(line.split()[-1])
        if vertex_count is None:
            raise ValueError(f"PLY missing vertex count: {path}")

        def prop_index(*names: str) -> int:
            for name in names:
                if name in properties:
                    return properties.index(name)
            raise ValueError(f"PLY missing required properties {names}: {path}")

        ix = prop_index("x")
        iy = prop_index("y")
        iz = prop_index("z")
        ir = prop_index("red", "r")
        ig = prop_index("green", "g")
        ib = prop_index("blue", "b")

        points_xyz: List[Tuple[float, float, float]] = []
        points_rgb: List[Tuple[int, int, int]] = []
        for _ in range(vertex_count):
            parts = f.readline().split()
            if len(parts) < len(properties):
                raise ValueError(f"Malformed PLY vertex row in {path}")
            points_xyz.append((float(parts[ix]), float(parts[iy]), float(parts[iz])))
            points_rgb.append((int(parts[ir]), int(parts[ig]), int(parts[ib])))
    return np.asarray(points_xyz, dtype=np.float64), np.asarray(points_rgb, dtype=np.uint8)


def source_points_from_scene(
    source_scene: Path, source_meta: dict, explicit_source_ply: Optional[Path]
) -> Tuple[np.ndarray, np.ndarray]:
    ply_candidates: List[Path] = []
    if explicit_source_ply is not None:
        ply_candidates.append(explicit_source_ply)
    if "ply_file_path" in source_meta:
        ply_candidates.append(source_scene / source_meta["ply_file_path"])
    for ply_path in ply_candidates:
        if ply_path.exists():
            return read_ascii_ply(ply_path)

    recon_dir = source_scene / "colmap" / "sparse" / "0"
    points_bin = recon_dir / "points3D.bin"
    if not points_bin.exists():
        raise FileNotFoundError(
            "Could not find a source sparse point cloud. Checked explicit/source transforms ply paths "
            f"and {points_bin}."
        )
    points_xyz, points_rgb = read_points3d_binary(points_bin)

    applied_transform = source_meta.get("applied_transform")
    if applied_transform is not None:
        applied = ensure_homogeneous(applied_transform)[:3, :]
        points_xyz = (applied[:3, :3] @ points_xyz.T).T + applied[:3, 3]
    return points_xyz, points_rgb


def write_ascii_ply(path: Path, points_xyz: np.ndarray, points_rgb: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_xyz)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uint8 red\n")
        f.write("property uint8 green\n")
        f.write("property uint8 blue\n")
        f.write("end_header\n")
        for xyz, rgb in zip(points_xyz, points_rgb):
            f.write(
                f"{float(xyz[0]):.8f} {float(xyz[1]):.8f} {float(xyz[2]):.8f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n"
            )


def rotation_error_degrees(target_rot: np.ndarray, source_rot: np.ndarray, world_rot: np.ndarray) -> float:
    delta = target_rot @ (world_rot @ source_rot).T
    trace = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def copy_scene_tree(source_scene: Path, output_scene: Path, overwrite: bool) -> None:
    if output_scene.exists():
        if not overwrite:
            raise FileExistsError(f"Output scene already exists: {output_scene}")
        shutil.rmtree(output_scene)
    shutil.copytree(
        source_scene,
        output_scene,
        ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints"),
    )


def main() -> int:
    args = parse_args()
    robot_scene = Path(args.robot_scene)
    sfm_scene = Path(args.sfm_scene)
    output_scene = Path(args.output_scene)
    source_ply = Path(args.source_ply) if args.source_ply is not None else None

    robot_transforms_path = robot_scene / "transforms.json"
    sfm_transforms_path = sfm_scene / "transforms.json"
    if not robot_transforms_path.exists():
        raise FileNotFoundError(f"Missing robot transforms.json: {robot_transforms_path}")
    if not sfm_transforms_path.exists():
        raise FileNotFoundError(f"Missing SfM transforms.json: {sfm_transforms_path}")

    robot_meta = read_json(robot_transforms_path)
    sfm_meta = read_json(sfm_transforms_path)
    frame_pairs = match_frames(
        sfm_meta["frames"],
        robot_meta["frames"],
        args.match_mode,
        args.source_sort,
        args.target_sort,
    )

    src_centers = np.asarray([camera_center_from_frame(src) for src, _ in frame_pairs], dtype=np.float64)
    tgt_centers = np.asarray([camera_center_from_frame(tgt) for _, tgt in frame_pairs], dtype=np.float64)
    scale, rotation, translation = umeyama_similarity(src_centers, tgt_centers)

    transformed_src_centers = apply_similarity(src_centers, scale, rotation, translation)
    center_errors = np.linalg.norm(transformed_src_centers - tgt_centers, axis=1)
    rot_errors = np.asarray(
        [
            rotation_error_degrees(rotation_from_frame(tgt), rotation_from_frame(src), rotation)
            for src, tgt in frame_pairs
        ],
        dtype=np.float64,
    )

    sparse_xyz, sparse_rgb = source_points_from_scene(sfm_scene, sfm_meta, source_ply)
    sparse_xyz_aligned = apply_similarity(sparse_xyz, scale, rotation, translation)

    copy_scene_tree(robot_scene, output_scene, overwrite=args.overwrite)

    out_transforms = read_json(output_scene / "transforms.json")
    out_transforms["ply_file_path"] = args.target_ply_name
    with open(output_scene / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(out_transforms, f, indent=4)

    write_ascii_ply(output_scene / args.target_ply_name, sparse_xyz_aligned, sparse_rgb)

    mapping_preview = [
        {
            "sfm_file": frame_basename(src),
            "robot_file": frame_basename(tgt),
        }
        for src, tgt in frame_pairs[: min(10, len(frame_pairs))]
    ]
    alignment_report = {
        "robot_scene": str(robot_scene),
        "sfm_scene": str(sfm_scene),
        "output_scene": str(output_scene),
        "match_mode": args.match_mode,
        "source_sort": args.source_sort,
        "target_sort": args.target_sort,
        "num_frame_pairs": len(frame_pairs),
        "scale": float(scale),
        "rotation_matrix": rotation.tolist(),
        "translation": translation.tolist(),
        "camera_center_rmse": rmse(center_errors),
        "camera_center_max_error": float(center_errors.max()),
        "rotation_error_mean_deg": float(rot_errors.mean()),
        "rotation_error_max_deg": float(rot_errors.max()),
        "mapping_preview": mapping_preview,
        "aligned_sparse_point_count": int(len(sparse_xyz_aligned)),
        "target_ply_file": args.target_ply_name,
    }
    with open(output_scene / "alignment.json", "w", encoding="utf-8") as f:
        json.dump(alignment_report, f, indent=4)

    print(f"Prepared robot-frame scene: {output_scene}")
    print(f"Matched camera pairs: {len(frame_pairs)}")
    print(f"Scale (sfm -> robot): {scale:.6f}")
    print(f"Camera center RMSE: {alignment_report['camera_center_rmse']:.6f}")
    print(f"Rotation error mean (deg): {alignment_report['rotation_error_mean_deg']:.4f}")
    print(f"Wrote aligned sparse point cloud: {output_scene / args.target_ply_name}")
    print(f"Wrote alignment report: {output_scene / 'alignment.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
