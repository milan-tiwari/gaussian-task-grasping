from pathlib import Path
import json

import matplotlib.pyplot as plt
import open3d as o3d
import trimesh as tr
from typing import Dict, Tuple

import viser
import viser.transforms as tf
import time

import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.pipelines.base_pipeline import Pipeline

from collections import deque
from scipy import ndimage
import copy
import torchvision
import kornia.morphology as kmorph
import kornia.filters as kfilters
import os


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}

class RealsenseCamera:
    # taken from graspnet demo parameters
    realsense = o3d.camera.PinholeCameraIntrinsic(
        1280, 720, 631.54864502, 631.20751953, 638.43517329, 366.49904066
    )

    @classmethod
    def get_camera(cls, c2w, center=None, image_shape=None, downscale=1) -> Cameras:
        if image_shape is None:
            height = cls.realsense.height
            width = cls.realsense.width
        else:
            height = image_shape[0]
            width = image_shape[1]

        if center is None:
            center_x = cls.realsense.intrinsic_matrix[0, 2]
            center_y = cls.realsense.intrinsic_matrix[1, 2]
        else:
            center_x = cls.realsense.intrinsic_matrix[0, 2] + (width-1)/2  - center[1] 
            center_y = cls.realsense.intrinsic_matrix[1, 2] + (height-1)/2 - center[0]

        camera = Cameras(
            camera_to_worlds = torch.Tensor(c2w).unsqueeze(0),
            fx = torch.Tensor([cls.realsense.intrinsic_matrix[0, 0]]),
            fy = torch.Tensor([cls.realsense.intrinsic_matrix[1, 1]]),
            cx = torch.Tensor([center_x]),
            cy = torch.Tensor([center_y]),
            width = torch.Tensor([width]).int(),
            height = torch.Tensor([height]).int(),
            )
        camera.rescale_output_resolution(downscale)

        return camera

class NerfstudioWrapper:
    def __init__(self, scene_path: str = None, pipeline: Pipeline = None):
        if scene_path is not None:
            _, pipeline, _, _ = eval_setup(Path(scene_path))
            pipeline.model.eval()
            self.pipeline = pipeline
        elif pipeline is not None:
            pipeline.model.eval()
            self.pipeline = pipeline
        else:
            raise ValueError("Must provide either scene_path or pipeline")
        self.camera_path: Cameras = pipeline.datamanager.train_dataset.cameras

        dp_outputs = pipeline.datamanager.train_dataparser_outputs
        self._dataparser_transform_h = np.concatenate(
            [dp_outputs.dataparser_transform.numpy(), np.array([[0, 0, 0, 1]])],
            axis=0,
        )
        applied_transform = np.eye(4)
        applied_transform[:3, :] = dp_outputs.dataparser_transform.numpy() #world to ns
        applied_transform = np.linalg.inv(applied_transform)
        applied_transform = applied_transform @ np.diag([1/dp_outputs.dataparser_scale]*3+[1]) #scale is post
        self.scene_dir = self._resolve_scene_dir(dp_outputs)
        self.scene_alignment, self.scene_alignment_path = self._discover_scene_alignment(self.scene_dir)
        self.scene_alignment_inv = np.linalg.inv(self.scene_alignment)
        self.applied_transform = self.scene_alignment @ applied_transform
        if self.scene_alignment_path is None:
            scene_label = self.scene_dir if self.scene_dir is not None else "<unknown scene>"
            print(
                f"[robot_lerf] No robot-frame alignment found for {scene_label}. "
                "Using identity transform."
            )
        else:
            print(f"[robot_lerf] Using robot-frame alignment: {self.scene_alignment_path}")

        self.num_cameras = len(self.camera_path.camera_to_worlds)

    @staticmethod
    def _resolve_scene_dir(dp_outputs) -> Path | None:
        image_filenames = getattr(dp_outputs, "image_filenames", None)
        if not image_filenames:
            return None
        try:
            first_image = Path(image_filenames[0]).resolve()
        except Exception:
            return None
        image_dir = first_image.parent
        if image_dir.name == "images":
            return image_dir.parent
        return image_dir

    @staticmethod
    def _similarity_matrix(scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = float(scale) * np.asarray(rotation, dtype=np.float64)
        transform[:3, 3] = np.asarray(translation, dtype=np.float64)
        return transform

    def _discover_scene_alignment(self, scene_dir: Path | None) -> Tuple[np.ndarray, Path | None]:
        identity = np.eye(4, dtype=np.float64)
        explicit_alignment = os.environ.get("ROBOT_LERF_ALIGNMENT_JSON")
        candidates: list[tuple[Path, bool]] = []
        if explicit_alignment:
            candidates.append((Path(explicit_alignment).expanduser(), True))
        if scene_dir is not None:
            scene_root = scene_dir.parent
            candidates.append((scene_root / "alignment.json", False))
            data_root = scene_root.parent
            for alignment_path in data_root.glob("*/alignment.json"):
                candidates.append((alignment_path, False))

        seen = set()
        for alignment_path, is_explicit in candidates:
            alignment_path = alignment_path.resolve()
            if alignment_path in seen or not alignment_path.is_file():
                continue
            seen.add(alignment_path)
            try:
                alignment = json.loads(alignment_path.read_text())
            except Exception:
                continue
            if scene_dir is not None and not is_explicit:
                sfm_scene = alignment.get("sfm_scene")
                if not sfm_scene:
                    continue
                if Path(sfm_scene).name != scene_dir.name:
                    continue
            if scene_dir is not None and is_explicit:
                sfm_scene = alignment.get("sfm_scene")
                if sfm_scene and Path(sfm_scene).name != scene_dir.name:
                    print(
                        f"[robot_lerf] Explicit alignment {alignment_path} targets {Path(sfm_scene).name}, "
                        f"but current scene is {scene_dir.name}. Using explicit override anyway."
                    )
            try:
                return self._similarity_matrix(
                    scale=float(alignment["scale"]),
                    rotation=np.asarray(alignment["rotation_matrix"], dtype=np.float64),
                    translation=np.asarray(alignment["translation"], dtype=np.float64),
                ), alignment_path
            except Exception:
                continue
        return identity, None

    # Note: applies to any real camera in world space
    def visercam_to_ns(self, c2w) -> np.ndarray:
        world_c2w = np.concatenate([c2w, np.array([[0, 0, 0, 1]])], axis=0)
        world_c2w = self.scene_alignment_inv @ world_c2w
        c2w = world_c2w[:3, :]
        c2w = self._dataparser_transform_h @ np.concatenate([c2w, np.array([[0, 0, 0, 1]])], axis=0)
        c2w = c2w[:3, :]
        c2w[:3, 2] *= -1
        c2w[:3, 1] *= -1
        return c2w

    def visercam_to_ns_world(self, c2w) -> np.ndarray:
        world_c2w = np.concatenate([c2w, np.array([[0, 0, 0, 1]])], axis=0)
        world_c2w = self.scene_alignment_inv @ world_c2w
        c2w = world_c2w[:3, :]
        c2w = self._dataparser_transform_h @ np.concatenate([c2w, np.array([[0, 0, 0, 1]])], axis=0)
        c2w = c2w[:3, :]
        return c2w

    # return in viser format
    def get_train_camera_c2w(self, train_cam_ind) -> np.ndarray:
        c2w = self.camera_path[train_cam_ind].camera_to_worlds.squeeze().numpy().copy()
        c2w[:3, 2] *= -1
        c2w[:3, 1] *= -1
        c2w = np.linalg.inv(self._dataparser_transform_h) @ np.concatenate([c2w, np.array([[0, 0, 0, 1]])], axis=0)
        c2w = self.scene_alignment @ c2w
        c2w = c2w[:3, :]
        return c2w

    def __call__(self, camera, render_lerf=False) -> Dict[str, np.ndarray]:
        if render_lerf:
            self.pipeline.model.step = 1000
        else:
            self.pipeline.model.step = 0
        camera_ray_bundle = camera.generate_rays(camera_indices=0).to(device)
        with torch.no_grad():
            outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        outputs['xyz'] = camera_ray_bundle.origins + camera_ray_bundle.directions * outputs['depth']
        for k, v in outputs.items():
            outputs[k] = v.squeeze().cpu().numpy()
        outputs['xyz'] = tr.transformations.transform_points(
            outputs['xyz'].reshape(-1, 3), 
            self.applied_transform
            ).reshape(outputs['xyz'].shape)
        return outputs

    # Helper function to rotate camera about a point along z axis
    def rotate_camera(self, curcam, rot, point):
        curcam = copy.deepcopy(curcam)
        world_to_point = torch.cat((torch.cat((torch.eye(3).to(device), -point.unsqueeze(1)), dim=1), torch.tensor([[0,0,0,1]]).to(device)), dim=0)
        rot = torch.tensor(rot)
        rotation_matrix_z = torch.tensor([[torch.cos(rot), -torch.sin(rot), 0, 0], [torch.sin(rot), torch.cos(rot), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).to(device)
        point_to_world = torch.inverse(world_to_point)
        homog_cam_to_world = torch.cat((curcam.camera_to_worlds.squeeze(), torch.tensor([[0,0,0,1]]).to(device)), dim=0)
        modcam_to_world = torch.matmul(point_to_world, torch.matmul(rotation_matrix_z, torch.matmul(world_to_point, homog_cam_to_world)))
        curcam.camera_to_worlds = modcam_to_world[:-1, :].unsqueeze(0)
        return curcam
    
    # Generate the lerf point cloud for the object
    def generate_lerf_pc(self, curcam, target_point):
        sweep = np.linspace(-np.pi/2,np.pi/2,6,dtype=np.float32)
        dino_dim = 384 # dimension of dino feature vector
        points = []
        rgbs = []
        dinos = []
        clips = []
        for i in sweep:
            mod_curcam = self.rotate_camera(curcam, i, target_point)
            with torch.no_grad():
                bundle = mod_curcam.generate_rays(camera_indices=0)
                outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(bundle)
            point = bundle.origins + bundle.directions * outputs["depth"]
            point = torch.reshape(point, (-1, 3))
            rgb = torch.reshape(outputs["rgb"], (-1, 3))
            dino = torch.reshape(outputs["dino"], (-1, dino_dim))
            points.append(point)
            rgbs.append(rgb)
            dinos.append(dino)
        points = torch.cat(points, dim=0)
        rgbs = torch.cat(rgbs, dim=0)
        dinos = torch.cat(dinos, dim=0).double().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.double().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(rgbs.double().cpu().numpy())
        raw_count = len(pcd.points)
        ind = np.arange(raw_count, dtype=np.int64)
        nb_neighbors = min(_env_int("ROBOT_LERF_DINO_OUTLIER_NB_NEIGHBORS", 20), max(raw_count - 1, 1))
        std_ratio = _env_float("ROBOT_LERF_DINO_OUTLIER_STD_RATIO", 1.0)
        min_keep = min(raw_count, _env_int("ROBOT_LERF_MIN_DINO_SWEEP_POINTS", 512))
        if raw_count > nb_neighbors and not _env_flag("ROBOT_LERF_DISABLE_DINO_OUTLIER", False):
            filtered_pcd, filtered_ind = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio,
            )
            if len(filtered_pcd.points) >= min_keep:
                pcd = filtered_pcd
                ind = np.asarray(filtered_ind, dtype=np.int64)
            else:
                print(
                    "[robot_lerf] DINO sweep outlier filter was too destructive; "
                    f"keeping unfiltered cloud ({len(filtered_pcd.points)}/{raw_count} kept)."
                )
        if ind is not None:
            dinos = dinos[ind]
        print(
            "[robot_lerf] LERF DINO sweep cloud: "
            f"raw={raw_count}, kept={len(pcd.points)}, outlier_std={std_ratio}"
        )
        return pcd, dinos, clips

    # Helper function to get DINO foreground mask and the object point in 2D
    def get_dino_first_comp_2d(self, curcam):
        default_view_cam = copy.deepcopy(curcam)
        
        # Hard-coded camera to world matrices for the zoomed in and overhead views of the experimental table setup
        camera_to_worlds_zoomed = torch.tensor([[
            [-2.9789e-02, -9.5959e-01,  2.7983e-01, -1.6356e-01],
            [ 9.9956e-01, -2.8598e-02,  8.3396e-03,  7.1693e-03],
            [ 5.5511e-17,  2.7995e-01,  9.6001e-01,  2.9950e-02]]],device=device)
        camera_to_worlds_overhead = torch.tensor([[
            [-0.0321, -0.8391,  0.5430,  0.0400 -0.05],
            [ 0.9995, -0.0270,  0.0175,  0.0134],
            [0.0000,  0.5433,  0.8395,  0.0789-0.05]]], device=device)

        # Find the first principal component of the DINO feature vectors
        default_view_cam.camera_to_worlds = camera_to_worlds_zoomed
        bundle = default_view_cam.generate_rays(camera_indices=0)
        outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(bundle)
        dinos = outputs["dino"]
        dinos = dinos.view(-1, dinos.shape[-1])
        _, _, v = torch.pca_lowrank(dinos, niter=5)
        dino_first_comp_2d = v[..., :1]
        
        # Use the first principal component to create a foreground mask
        default_view_cam.camera_to_worlds = camera_to_worlds_overhead
        bundle = default_view_cam.generate_rays(camera_indices=0)
        outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(bundle)
        dinos = outputs["dino"]
        dinos = dinos.view(-1, dinos.shape[-1])
        THRESHOLD = 0.8 # Threshold to create the DINO mask depending if the objts are in the foreground or background
        mask = torch.where(torch.matmul(dinos, dino_first_comp_2d) < THRESHOLD, 1, 0)
        mask_test=torch.reshape(mask, (curcam.height.item(), curcam.width.item())).cpu().numpy()
        labels, _=ndimage.label(mask_test)
        component_sizes = np.bincount(labels.ravel())
        largets_component = np.argmax(component_sizes[1:]) + 1
        largets_component_area = component_sizes[largets_component]
        if largets_component_area > 0.5 * mask.shape[0] * mask.shape[1]:
            dino_first_comp_2d = -1 * dino_first_comp_2d
            mask = torch.where(torch.matmul(dinos, dino_first_comp_2d) < THRESHOLD, 1, 0)
        relevancy_0 = outputs["relevancy_0"]
        relevancy_0 = relevancy_0.view(-1, relevancy_0.shape[-1])
        masked_relevancy = relevancy_0 * mask
        masked_relevancy = masked_relevancy.view(curcam.height, curcam.width, 1).permute(2, 0, 1).unsqueeze(0)
        blur = kfilters.BoxBlur((3, 3))
        masked_relevancy = blur(masked_relevancy)[0].permute(1, 2, 0)
        target_idx = torch.topk(masked_relevancy.squeeze().flatten(), 1, largest=True).indices.item() #change for multiple
        all_points = bundle.origins + bundle.directions * outputs["depth"]
        all_points = torch.reshape(all_points, (-1, 3))
        target_points = all_points[target_idx]
        return dino_first_comp_2d, target_points
    
    # Helper function to project a point cloud to image space
    def project_to_image(self, curcam, point_cloud, round_px=True):
        K = curcam.get_intrinsics_matrices().to(device)[0]
        points_proj = torch.matmul(K, point_cloud)
        if len(points_proj.shape) == 1:
            points_proj = points_proj[:, None]
        point_depths = points_proj[2, :]
        point_z = point_depths.repeat(3, 1)
        points_proj = torch.divide(points_proj, point_z)
        if round_px:
            points_proj = torch.round(points_proj)
        points_proj = points_proj[:2, :].int()

        valid_ind = torch.where((points_proj[0, :] >= 0) & (points_proj[1, :] >= 0) & (points_proj[0, :] < curcam.image_width.item()) & (points_proj[1, :] < curcam.image_height.item()), 1, 0).to(device)
        valid_ind = torch.argwhere(valid_ind.squeeze())
        depth_data = torch.ones([curcam.image_height, curcam.image_width], device=device) * -1
        depth_data[points_proj[1, valid_ind], points_proj[0, valid_ind]] = point_depths[valid_ind]
        reverse = valid_ind.flip(0)
        depth_data[points_proj[1, reverse], points_proj[0, reverse]] = torch.minimum(point_depths[reverse], depth_data[points_proj[1, reverse], points_proj[0, reverse]])
        return depth_data

    # Flood fill a point cloud given seed points
    def flood_fill_3d(self, pcd, pcd_tree, dino_vectors, seed_indices, tolerance):
        radius = _env_float("ROBOT_LERF_DINO_FLOOD_RADIUS", 0.05)
        q = deque(seed_indices)
        seed_value = dino_vectors[seed_indices].mean(axis=0)
        np_pts = np.asarray(pcd.points)
        mask = np.zeros(np_pts.shape[0], dtype=bool)
        count = 0
        while q:
            pt_indx = q.popleft()
            if mask[pt_indx] == False:
                mask[pt_indx] = True
                [_, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[pt_indx], radius)
                np_idx = np.asarray(idx)
                idx_tolerance = np.mean((dino_vectors[np_idx]- seed_value) ** 2, axis=-1)
                idx_of_id = np.argwhere(idx_tolerance < tolerance).squeeze()
                temp_list = np_idx[idx_of_id]
                added = temp_list.tolist() if isinstance(temp_list, np.ndarray) else [temp_list]
                q.extend(added)
                count += 1
        return mask
    
    # Main function to get generate the lerf point cloud for the object part
    def get_lerf_pointcloud(self, curcam, render_lerf=True) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        if render_lerf:
            self.pipeline.model.step = 1000
        else:
            self.pipeline.model.step = 0
        curcam = curcam.to(device)
        dino_first_comp_2d, target_points = self.get_dino_first_comp_2d(curcam)
        
        dino_first_comp_2d = dino_first_comp_2d.double().cpu().detach().numpy()
        if curcam is None:
            return
        px, py, _ = target_points
        
        # Hard-coded camera to world matrices specific to experimental table setup
        X_OFFSET, Y_OFFSET = 0.7, 0
        curcam.camera_to_worlds = torch.tensor([[
            [-0.1114, -0.4216,  0.8999,  px.item()+X_OFFSET],
            [ 0.9938, -0.0472,  0.1008,  py.item()+Y_OFFSET],
            [ 0.0000,  0.9056,  0.4242, -0.20]]], device=device)
        
        pcd, dinos, _ = self.generate_lerf_pc(curcam, target_points)
        raw_dino_count = len(pcd.points)
        if raw_dino_count == 0:
            print("[robot_lerf] LERF DINO sweep produced zero points.")
            return None, None, None, None, None
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        [_, idx, _] = pcd_tree.search_knn_vector_3d(target_points.double().cpu().detach().numpy(), 1)
        seed_indices = np.array(idx)
        dinos = np.matmul(dinos, dino_first_comp_2d)
        flood_tolerance = _env_float("ROBOT_LERF_DINO_FLOOD_TOLERANCE", 7.5)
        print(
            "[robot_lerf] floodfill start: "
            f"seed_count={len(seed_indices)}, candidates={raw_dino_count}, tolerance={flood_tolerance}"
        )
        mask = self.flood_fill_3d(pcd, pcd_tree, dinos, seed_indices, tolerance=flood_tolerance)
        min_dino_points = min(raw_dino_count, _env_int("ROBOT_LERF_MIN_DINO_POINTS", 64))
        if int(mask.sum()) < min_dino_points:
            _, fallback_idx, _ = pcd_tree.search_knn_vector_3d(
                target_points.double().cpu().detach().numpy(),
                min_dino_points,
            )
            fallback_mask = np.zeros(raw_dino_count, dtype=bool)
            fallback_mask[np.asarray(fallback_idx, dtype=np.int64)] = True
            print(
                "[robot_lerf] DINO floodfill was too sparse; "
                f"expanding from {int(mask.sum())} to {int((mask | fallback_mask).sum())} nearest points."
            )
            mask = mask | fallback_mask
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[mask, :])
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask, :])
        print(f"[robot_lerf] LERF DINO object cloud after floodfill: {len(pcd.points)}")
        dino_pcd = copy.deepcopy(pcd)
        points, rgbs, relevancies = [], [], []
        fallback_points, fallback_rgbs, fallback_relevancies = [], [], []
        
        sweep= np.linspace(-np.pi/2,np.pi/2,6,dtype=np.float32)
        for i in sweep:
            mod_curcam = self.rotate_camera(curcam, i, target_points)
            homog_cam_to_world = torch.cat((mod_curcam.camera_to_worlds.squeeze(), torch.tensor([[0,0,0,1]]).to(device)), dim=0)
            homog_world_to_cam = torch.inverse(homog_cam_to_world)
            pcd_points = np.asarray(pcd.points)
            homog_pcd_points = np.ones((pcd_points.shape[0],4))
            homog_pcd_points[:, :3] = pcd_points
            rotation_matrix_x = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], device=device).float() #180 deg rot about cam x axis
            point_cloud_in_camera_frame = torch.matmul(rotation_matrix_x, torch.matmul(homog_world_to_cam, torch.tensor(homog_pcd_points.T, device=device).float()))
            dino_2d_data = self.project_to_image(mod_curcam, point_cloud_in_camera_frame[:3, :])
            dino_2d_mask = torch.where(dino_2d_data > 0, 1, 0)
            image_point = dino_2d_mask.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).float()
            kernel = torch.ones(3, 3).to(device)
            dino_2d_mask = kmorph.dilation(image_point, kernel)
            dino_2d_mask = dino_2d_mask.squeeze()[0, :, :]
            with torch.no_grad():
                bundle = mod_curcam.generate_rays(camera_indices=0)
                s = (*bundle.shape,1)
                bundle.nears = torch.full(s , 0.05, device=device)
                bundle.fars = torch.full(s, 10, device=device)
                outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(bundle)
            # testing if this works:
            lerf_mask = torch.where((outputs['relevancy_1'] > 0) & (dino_2d_mask.unsqueeze(2) > 0) & (torch.abs(outputs['depth'].squeeze() - dino_2d_data).unsqueeze(2) < 0.3), 1, 0)
            # ACTUAL:
            # lerf_mask = torch.where((outputs['relevancy_1'] > 0) & (torch.abs(outputs['depth'].squeeze() - dino_2d_data).unsqueeze(2) < 0.08), 1, 0)
            point = bundle.origins + bundle.directions * outputs["depth"]
            point = torch.reshape(point, (-1, 3))
            rgb = torch.reshape(outputs["rgb"], (-1, 3))
            relevancy = torch.reshape(outputs["relevancy_1"], (-1, 1))
            lerf_mask = torch.reshape(lerf_mask, (-1, 1)).squeeze()
            mask_indices = torch.nonzero(lerf_mask, as_tuple=False).reshape(-1)
            points.append(point[mask_indices])
            rgbs.append(rgb[mask_indices])
            relevancies.append(relevancy[mask_indices])

            fallback_topk = _env_int("ROBOT_LERF_RELAXED_TOPK_PER_VIEW", 256)
            if fallback_topk > 0:
                rel_flat = relevancy.squeeze(-1)
                finite_indices = torch.nonzero(torch.isfinite(rel_flat), as_tuple=False).reshape(-1)
                positive_indices = finite_indices[rel_flat[finite_indices] > 0]
                candidate_indices = positive_indices if positive_indices.numel() > 0 else finite_indices
                if candidate_indices.numel() > 0:
                    k = min(fallback_topk, candidate_indices.numel())
                    top_local = torch.topk(rel_flat[candidate_indices], k, largest=True).indices
                    fallback_indices = candidate_indices[top_local]
                    fallback_points.append(point[fallback_indices])
                    fallback_rgbs.append(rgb[fallback_indices])
                    fallback_relevancies.append(relevancy[fallback_indices])

        points = torch.cat(points, dim=0)
        rgbs = torch.cat(rgbs, dim=0)
        relevancies = torch.cat(relevancies, dim=0)
        min_semantic_points = _env_int("ROBOT_LERF_MIN_SEMANTIC_POINTS", 64)
        if len(relevancies) < min_semantic_points and fallback_relevancies:
            relaxed_points = torch.cat(fallback_points, dim=0)
            relaxed_rgbs = torch.cat(fallback_rgbs, dim=0)
            relaxed_relevancies = torch.cat(fallback_relevancies, dim=0)
            if len(relaxed_relevancies) > len(relevancies):
                print(
                    "[robot_lerf] Strict LERF+DINO mask was too sparse; "
                    f"using relaxed LERF top activations ({len(relevancies)} -> {len(relaxed_relevancies)} points)."
                )
                points, rgbs, relevancies = relaxed_points, relaxed_rgbs, relaxed_relevancies
        if len(relevancies) == 0:
            print("No points found with positive lerf relevancy")
            return None, None, None, None, None
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.double().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(rgbs.double().cpu().numpy())
        raw_semantic_count = len(pcd.points)
        ind = np.arange(raw_semantic_count, dtype=np.int64)
        nb_neighbors = min(_env_int("ROBOT_LERF_LERF_OUTLIER_NB_NEIGHBORS", 12), max(raw_semantic_count - 1, 1))
        std_ratio = _env_float("ROBOT_LERF_LERF_OUTLIER_STD_RATIO", 1.0)
        min_keep = min(raw_semantic_count, _env_int("ROBOT_LERF_MIN_SEMANTIC_POINTS", 64))
        if raw_semantic_count > nb_neighbors and not _env_flag("ROBOT_LERF_DISABLE_LERF_OUTLIER", False):
            filtered_pcd, filtered_ind = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio,
            )
            if len(filtered_pcd.points) >= min_keep:
                pcd = filtered_pcd
                ind = np.asarray(filtered_ind, dtype=np.int64)
            else:
                print(
                    "[robot_lerf] Semantic outlier filter was too destructive; "
                    f"keeping unfiltered cloud ({len(filtered_pcd.points)}/{raw_semantic_count} kept)."
                )
        relevancies = relevancies[torch.as_tensor(ind, device=relevancies.device, dtype=torch.long)]
        rel_min = relevancies.min()
        rel_span = relevancies.max() - rel_min
        if float(rel_span.detach().cpu()) > 1e-8:
            relevancies = (relevancies - rel_min) / rel_span
        else:
            print("[robot_lerf] LERF semantic relevancy was flat; using uniform semantic weights.")
            relevancies = torch.ones_like(relevancies)
        print(
            "[robot_lerf] LERF semantic cloud: "
            f"raw={raw_semantic_count}, kept={len(pcd.points)}, outlier_std={std_ratio}"
        )
        return pcd, relevancies.cpu().numpy(), dino_pcd, target_points, None

    def create_pointcloud(self) -> Tuple[tr.PointCloud, np.ndarray]:
        self.pipeline.model.step = 0
        if hasattr(self.pipeline.datamanager, "cached_train"):
            global_pointcloud = self._create_pointcloud_from_full_images(num_points=1000000)
        else:
            try:
                from nerfstudio.exporter.exporter_utils import generate_point_cloud
            except ImportError as exc:
                raise RuntimeError(
                    "Fell back to nerfstudio.exporter.generate_point_cloud(), but that import failed. "
                    "This usually means the optional pymeshlab dependency is broken in the current "
                    "environment. Use the FullImageDatamanager path when available, or repair the "
                    "pymeshlab/Qt install for this env."
                ) from exc
            pixel_sampler = getattr(self.pipeline.datamanager, "train_pixel_sampler", None)
            orig_num_rays_per_batch = None
            if pixel_sampler is not None and hasattr(pixel_sampler, "num_rays_per_batch"):
                orig_num_rays_per_batch = pixel_sampler.num_rays_per_batch
                pixel_sampler.num_rays_per_batch = 100000

            global_pointcloud = generate_point_cloud(
                self.pipeline,
                remove_outliers=True,
                std_ratio=0.1,
                depth_output_name='depth',
                num_points=1000000,
            )
            if pixel_sampler is not None and orig_num_rays_per_batch is not None:
                pixel_sampler.num_rays_per_batch = orig_num_rays_per_batch
        global_pointcloud.points = o3d.utility.Vector3dVector(tr.transformations.transform_points(np.asarray(global_pointcloud.points), self.applied_transform)) #nerfstudio pc to world/viser pc
        global_pointcloud = global_pointcloud.voxel_down_sample(0.001)

        #find the table by fitting a plane
        # _,plane_ids = global_pointcloud.segment_plane(.003,3,1000)
        # plane_pts = global_pointcloud.select_by_index(plane_ids)
        # #downsample the points to reduce density
        # plane_pts = plane_pts.voxel_down_sample(0.001)
        # #remove outlier points from the plane to restrict it to the table points
        # plane_pts,_=plane_pts.remove_radius_outlier(150,.01,print_progress=True)
        # #get the oriented bounding box of this restricted plane (should be the table)
        # bbox_o3d = plane_pts.get_oriented_bounding_box()
        # #stretch the 3rd dimension, which corresponds to raising the bounding box
        # stretched_extent = np.array((bbox_o3d.extent[0],bbox_o3d.extent[1],1))
        # inflated_bbox = o3d.geometry.OrientedBoundingBox(bbox_o3d.center,bbox_o3d.R,stretched_extent)

        # it's probably more robust to just hardcode the table
        table_center = np.array((.45,0,-.18))
        inflated_bbox = o3d.geometry.OrientedBoundingBox(table_center,np.eye(3),np.array((.5,.7,1)))
        # inv_t = self.applied_transform
        # inflated_bbox = inflated_bbox.rotate(inv_t[:3,:3])
        # inflated_bbox = inflated_bbox.translate(inv_t[:3,3])
        world_pointcloud_o3d = global_pointcloud.crop(inflated_bbox)
        if len(world_pointcloud_o3d.points) == 0:
            alignment_hint = (
                f"using alignment file {self.scene_alignment_path}"
                if self.scene_alignment_path is not None
                else "with no alignment file loaded"
            )
            raise RuntimeError(
                "Robot-frame crop produced zero points. "
                f"The scene was loaded {alignment_hint}. "
                "For HLOC checkpoints, set ROBOT_LERF_ALIGNMENT_JSON to the robot-frame alignment.json "
                "before launching gen_grasp.py."
            )
        world_pointcloud = tr.PointCloud(
            vertices=np.asarray(world_pointcloud_o3d.points),
            colors=np.asarray(world_pointcloud_o3d.colors)
        )
        global_pointcloud = tr.PointCloud(
            vertices=np.asarray(global_pointcloud.points),
            colors=np.asarray(global_pointcloud.colors)
        )
        return world_pointcloud,global_pointcloud,table_center

    def _create_pointcloud_from_full_images(self, num_points: int = 1000000) -> o3d.geometry.PointCloud:
        """Create a point cloud for camera-based datamanagers such as Splatfacto's FullImageDatamanager."""
        train_cameras = getattr(self.pipeline.datamanager, "train_cameras", None)
        if train_cameras is None:
            train_cameras = self.pipeline.datamanager.train_dataset.cameras
        train_cameras = train_cameras.to(device)
        num_cameras = len(train_cameras)
        samples_per_camera = max(4096, int(np.ceil(float(num_points) / max(num_cameras, 1))))

        points = []
        rgbs = []
        for cam_idx in range(num_cameras):
            camera = copy.deepcopy(train_cameras[cam_idx : cam_idx + 1]).to(device)
            if camera.metadata is None:
                camera.metadata = {}
            else:
                camera.metadata = dict(camera.metadata)
            camera.metadata["cam_idx"] = cam_idx

            with torch.no_grad():
                outputs = self.pipeline.model.get_outputs_for_camera(camera)

            ray_bundle = camera.generate_rays(camera_indices=0)
            point = ray_bundle.origins + ray_bundle.directions * outputs["depth"]
            rgb = outputs["rgb"]

            accumulation = outputs.get("accumulation")
            if accumulation is not None:
                mask = accumulation.squeeze(-1) > 0.5
            else:
                mask = torch.isfinite(outputs["depth"].squeeze(-1)) & (outputs["depth"].squeeze(-1) > 0)

            point = point[mask]
            rgb = rgb[mask]
            if point.shape[0] == 0:
                continue

            if point.shape[0] > samples_per_camera:
                choice = torch.randperm(point.shape[0], device=point.device)[:samples_per_camera]
                point = point[choice]
                rgb = rgb[choice]

            points.append(point.detach().cpu())
            rgbs.append(rgb.detach().cpu())

        if not points:
            raise RuntimeError("Point cloud generation produced no valid points from the training cameras.")

        points_tensor = torch.cat(points, dim=0)
        rgbs_tensor = torch.cat(rgbs, dim=0)
        if points_tensor.shape[0] > num_points:
            choice = torch.randperm(points_tensor.shape[0])[:num_points]
            points_tensor = points_tensor[choice]
            rgbs_tensor = rgbs_tensor[choice]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_tensor.double().numpy())
        pcd.colors = o3d.utility.Vector3dVector(rgbs_tensor.double().numpy())
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.1)
        return pcd
