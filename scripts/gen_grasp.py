import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BUILD_MARKER = "gen_grasp_build_2026_04_19_scores_guard_v1"

import time
import traceback
import numpy as np
import trimesh as tr
import tyro

import os
import os.path as osp
import open3d as o3d
import matplotlib
from typing import List, Dict, Optional, Tuple

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import viser
import viser.transforms as tf
from autolab_core import RigidTransform

from graspnetAPI import GraspGroup, Grasp
from nerfstudio.pipelines.base_pipeline import Pipeline

from robot_lerf.graspnet_baseline.graspnet_module import GraspNetModule
from robot_lerf.capture_utils import _generate_hemi
from robot_lerf.grasp_planner_cmk import UR5GraspPlanner
from robot_lerf.scene_backends import (
    RealsenseCamera,
    SceneBackend,
    backend_runtime_error,
    create_scene_backend,
    training_method_name_for_backend,
)


def _pointcloud_colors(pointcloud: tr.PointCloud) -> np.ndarray:
    colors = getattr(pointcloud.visual, "vertex_colors", None)
    if colors is not None and len(colors) == len(pointcloud.vertices):
        return np.asarray(colors[:, :3])
    raw_colors = getattr(pointcloud, "colors", None)
    if raw_colors is not None and len(raw_colors) == len(pointcloud.vertices):
        raw_colors = np.asarray(raw_colors)
        if raw_colors.ndim == 2 and raw_colors.shape[1] >= 3:
            return raw_colors[:, :3]
    return np.repeat(np.array([[0.7, 0.7, 0.7]], dtype=np.float32), len(pointcloud.vertices), axis=0)


def _subsample_points_for_display(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points)
    colors = np.asarray(colors)
    if len(points) <= max_points:
        return points, colors
    selection = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[selection], colors[selection]


def get_relevancy_pointcloud(scene_backend: SceneBackend, **kwargs):
    """Get relevancy pointcloud, used to get semantic score

    Args:
        scene_backend (SceneBackend): scene representation used for querying

    Returns:
        o3d.utility.Vector3DVector: points in pointcloud (xyz)
        np.ndarray: relevancy score
    """
    lerf_xyz, lerf_relevancy = [], []
    center_pos_matrix = np.array([[ 1., 0., 0., 0.45], [0., -0.70710678,  0.70710678, -0.28284271],[ 0, -0.70710678, -0.70710678,  0.10284271]])
    c2w = scene_backend.visercam_to_model(center_pos_matrix)
    rscam = RealsenseCamera.get_camera(c2w, downscale=1/4)
    lerf_pcd, lerf_relevancy, dino_pcd, pick_object_pt, place_pt = scene_backend.get_lerf_pointcloud(rscam)
    pick_object_pt = tr.transformations.transform_points(pick_object_pt.unsqueeze(0).cpu().numpy(), scene_backend.applied_transform).squeeze()
    if place_pt is not None:
        place_pt = tr.transformations.transform_points(place_pt.unsqueeze(0).cpu().numpy(), scene_backend.applied_transform).squeeze()
    lerf_pcd.points = o3d.utility.Vector3dVector(tr.transformations.transform_points(np.asarray(lerf_pcd.points), scene_backend.applied_transform)) # model pc to world/viser pc
    dino_pcd.points = o3d.utility.Vector3dVector(tr.transformations.transform_points(np.asarray(dino_pcd.points), scene_backend.applied_transform))
    lerf_points_o3d = lerf_pcd.points
    dino_points_o3d = dino_pcd.points

    return lerf_points_o3d, lerf_relevancy, dino_points_o3d, pick_object_pt, place_pt

def get_grasps(
    graspnet: GraspNetModule,
    world_pointcloud: tr.PointCloud,
    hemisphere: List[RigidTransform],
    graspnet_batch_size: int = 15,
    ) -> GraspGroup:
    """Get grasps from graspnet, as images taken from the hemisphere
    
    Args: 
        graspnet (GraspNetModule): graspnet module
        world_pointcloud (tr.PointCloud): world pointcloud
        hemisphere (List[RigidTransform]): list of camera poses
    
    Returns:
        GraspGroup: grasps
    """
    torch.cuda.empty_cache()
    gg_all = None
    for i in range(0, len(hemisphere), graspnet_batch_size):
        start = time.time()
        ind_range = range(i, min(i+graspnet_batch_size, len(hemisphere)))
        rgbd_cropped_list = []
        for j in ind_range:
            c2w = hemisphere[j].matrix[:3,:]
            rgbd_cropped = world_pointcloud.copy()
            rgbd_cropped.vertices = tr.transformations.transform_points(
                rgbd_cropped.vertices,
                np.linalg.inv(np.concatenate([c2w, [[0, 0, 0, 1]]], axis=0))
            )
            rgbd_cropped_list.append(rgbd_cropped)
        print("Transform time: ", time.time() - start)

        gg_list = graspnet(rgbd_cropped_list)
        for g_ind, gg in enumerate(gg_list):
            c2w = hemisphere[i + g_ind].matrix[:3,:]
            gg.transform(np.concatenate([c2w, [[0, 0, 0, 1]]], axis=0))
        print(f"Grasp pred time: {time.time() - start:.2f}s")
        start = time.time()

        gg_all_curr = gg_list[0]
        for gg in gg_list[1:]:
            gg_all_curr.add(gg)
        gg = gg_all_curr

        # If the grasps are too close to the ground, then lift them a bit.
        # This is hardcoded though, so it might not work for all scenes
        gg_translations = gg.translations
        gg_translations[gg_translations[:, 2] < -0.16] += np.tile(np.array([0, 0, 0.01]), ((gg_translations[:, 2] < -0.16).sum(), 1))
        gg.translations = gg_translations
        # gg[gg.translations[:, 2] < -0.16].translations += np.tile(np.array([0, 0, 0.04]), ((gg.translations[:, 2] < -0.16).sum(), 1))
        gg = gg[(gg.translations[:, 0] > 0.22)] #& (gg.translations[:, 2] < 0.05)]

        gg = gg[np.abs(gg.rotation_matrices[:, :, 1][:, 2]) < 0.5]

        # gg = gg[gg.scores > 0.6]
        if len(gg) == 0:
            continue

        gg = gg.nms(translation_thresh=0.01, rotation_thresh=30.0/180.0*np.pi)

        # select grasps that are not too close to the table
        # Currently, this function does general grasp filtering (using collision detection, grasp includes non-table components, ...)
        gg = graspnet.local_collision_detection(gg)

        print(f"Collision detection time: {time.time() - start:.2f}s")
        print(f"Post proc time: {time.time() - start:.2f}s")
        if gg_all is None:
            gg_all = gg
        else:
            gg_all.add(gg)

    if gg_all is None:
        return GraspGroup()
    
    gg_all = gg_all.nms(translation_thresh=0.01, rotation_thresh=30.0/180.0*np.pi)
    gg_all.sort_by_score()
    torch.cuda.empty_cache()

    return gg_all

def instantiate_scene_from_model(
    server: viser.ViserServer,
    graspnet_ckpt: str,
    floor_height: float,
    scene_name: str,
    backend_type: str = "nerfstudio",
    config_path: str = None,
    pipeline: Pipeline = None,
) -> Tuple[SceneBackend, tr.PointCloud, tr.PointCloud, np.ndarray, GraspNetModule, GraspGroup, np.ndarray]:
    """Instantiate a scene backend and its associated GraspNetModule.

    Args:
        server (viser.ViserServer): viser server, used for visualization
        config_path (str): model config path
        graspnet_ckpt (str): GraspNet checkpoint path
        floor_height (float): floor height

    Returns:
        Tuple[NerfstudioWrapper, tr.PointCloud, tr.PointCloud, np.ndarray, GraspNetModule]:
         - world_pointcloud: world pointcloud
         - global_pointcloud: nerf pointcloud (not cropped)
         - table_center: table center
         - graspnet: graspnet module
         - grasps: GraspGroup
         - overall_scores: overall scores

    """
    scene_backend = create_scene_backend(
        backend_type=backend_type,
        scene_path=config_path,
        pipeline=pipeline,
    )

    world_pointcloud, global_pointcloud, table_center = scene_backend.create_pointcloud()

    graspnet = GraspNetModule()
    graspnet.init_net(graspnet_ckpt, global_pointcloud, cylinder_radius=0.04, floor_height=floor_height)

    global_colors = _pointcloud_colors(global_pointcloud)
    workspace_colors = _pointcloud_colors(world_pointcloud)
    global_display_max = int(os.environ.get("ROBOT_LERF_GLOBAL_DISPLAY_POINTS", "180000"))
    workspace_display_max = int(os.environ.get("ROBOT_LERF_WORKSPACE_DISPLAY_POINTS", "80000"))
    global_display_points, global_display_colors = _subsample_points_for_display(
        np.asarray(global_pointcloud.vertices),
        global_colors,
        global_display_max,
    )
    workspace_display_points, workspace_display_colors = _subsample_points_for_display(
        np.asarray(world_pointcloud.vertices),
        workspace_colors,
        workspace_display_max,
    )
    print(
        "[robot_lerf] Visualizing scene point clouds: "
        f"global={len(global_pointcloud.vertices)} pts "
        f"(displaying {len(global_display_points)}), "
        f"workspace={len(world_pointcloud.vertices)} pts "
        f"(displaying {len(workspace_display_points)})"
    )
    if len(global_display_points) > 0:
        server.add_point_cloud(
            name="/world_pointcloud",
            points=global_display_points,
            colors=global_display_colors,
            point_size=0.002,
        )
    if len(workspace_display_points) > 0:
        server.add_point_cloud(
            name="/workspace_pointcloud",
            points=workspace_display_points,
            colors=workspace_display_colors,
            point_size=0.003,
        )

    # server.add_point_cloud(
    #     name=f"/coll_pointcloud",
    #     points=graspnet.pointcloud_vertices,
    #     colors=np.repeat(np.array([[0, 1, 0]]), len(graspnet.pointcloud_vertices), axis=0),
    #     point_size=0.002,
    #     visible=False
    # )
    # world_pointcloud.export("world_pointcloud.ply")

    hemi_radius = 2
    hemi_theta_N = 15
    hemi_phi_N = 15
    hemi_th_range = 90
    hemi_phi_down = 0
    hemi_phi_up = 70

    if osp.exists(f"outputs/{scene_name}/grasps.npy"):
        grasps = GraspGroup(np.load(f"outputs/{scene_name}/grasps.npy"))
    else:
        grasp_hemisphere = _generate_hemi(
            hemi_radius,hemi_theta_N,hemi_phi_N,
            (np.deg2rad(-hemi_th_range),np.deg2rad(hemi_th_range)),
            (np.deg2rad(hemi_phi_down),np.deg2rad(hemi_phi_up)),
            center_pos=table_center,look_pos=table_center
            )
        grasps = get_grasps(graspnet, world_pointcloud, grasp_hemisphere)
        grasps.save_npy(f"outputs/{scene_name}/grasps.npy")

    # # just add all the grasps lol
    # for i, grasp in enumerate(grasps):
    #     add_grasps(server, grasp, i, 0.5)

    return (
        scene_backend,
        world_pointcloud,
        global_pointcloud,
        table_center,
        grasps,
        np.array(grasps.scores)
    )

def add_grasps(
    server: viser.ViserServer,
    grasp: Grasp,
    ind: int,
    score: float,
) -> Tuple[viser.SceneNodeHandle, viser.SceneNodeHandle, viser.SceneNodeHandle, viser.SceneNodeHandle]:
    """Curry function for adding grasps to the scene.

    Args:
        server (viser.ViserServer): _description_
        grasp (Grasp): _description_
        ind (int): _description_
        score (float): The color is based on the score -- put it from 0 to 1

    Returns:
        Tuple[viser.SceneNodeHandle, viser.SceneNodeHandle, viser.SceneNodeHandle]:
         - frame_handle: [Grasp frame (graspnetAPI)] to [world]
         - grasp_handle: mesh
         - ur5_handle: [UR5 frame (EE)] to [Grasp frame (graspnetAPI)]
    """
    colormap = matplotlib.colormaps['RdYlGn']
    robot_frame_R = RigidTransform(
        rotation=RigidTransform.y_axis_rotation(np.pi/2) @ RigidTransform.z_axis_rotation(np.pi/2)
    )

    default_grasp = Grasp()
    default_grasp.depth = grasp.depth
    default_grasp.width = grasp.width
    default_grasp.height = grasp.height
    default_grasp = default_grasp.to_open3d_geometry()

    frame_handle = server.add_frame(
        name=f'/lerf/grasps_{ind}',
        wxyz=tf.SO3.from_matrix(grasp.rotation_matrix).wxyz,
        position=grasp.translation,
        show_axes=False
    )
    grasp_handle = server.add_mesh(
        name=f'/lerf/grasps_{ind}/mesh',
        vertices=np.asarray(default_grasp.vertices),
        faces=np.asarray(default_grasp.triangles),
        color=colormap(score)[:3],
    )
    ur5_handle = server.add_frame(
        name=f'/lerf/grasps_{ind}/ur5',
        wxyz=robot_frame_R.quaternion,
        # position=np.array([0.03, 0, 0]),
        position=np.array([grasp.depth-0.015, 0, 0]),
        axes_length=0.05,
        axes_radius=0.002,
        show_axes=True,
        visible=False
    )
    return frame_handle, grasp_handle, ur5_handle


def get_bbox_from_grasp(gg: Grasp) -> o3d.geometry.OrientedBoundingBox:
    center = gg.translation
    R = gg.rotation_matrix
    H= np.eye(4)
    H[:3,:3] = R
    H[:3,3] = center
    extent=np.array((gg.depth,gg.width,gg.height))
    box = o3d.geometry.OrientedBoundingBox(center,H[:3,:3],extent)
    return box


def main(
    config_path: Optional[str] = None,  # Nerfstudio model config path, of format outputs/.../config.yml; if None, make sure you capture!
    graspnet_ckpt: str = 'robot_lerf/graspnet_baseline/logs/log_kn/checkpoint.tar',  # GraspNet checkpoint path
    scene_backend: str = "nerfstudio",
    viser_port: int = 8080,
    viser_host: str = "0.0.0.0",
    ):

    print(f"[robot_lerf] Starting {BUILD_MARKER}")
    server = viser.ViserServer(host=viser_host, port=viser_port)
    
    curr_client = None
    @server.on_client_connect
    def set_init_cam(client: viser.ClientHandle) -> None:
        nonlocal curr_client
        curr_client = client
        try:
            client.camera.position = np.array([0.87, 0.19, 0.24])
            client.camera.look_at = np.array([0.42, 0.066, -0.185])
        except Exception:
            # Camera initialization is a convenience only; older/newer viser
            # builds should still allow the app to load even if this API changes.
            pass

    # Create all necessary global variables
    grasps, grasps_dict, lerf_scores, geom_scores, overall_scores, pick_object_pt, place_point, grasp_point, fin_grasp = None, {}, [], [], [], None, None, None, None
    lerf_points_o3d, lerf_relevancy = None, None
    runtime_error = backend_runtime_error(scene_backend)
    if runtime_error is not None and config_path is not None:
        raise RuntimeError(runtime_error)
    if config_path is not None:
        scene_model, world_pointcloud, global_pointcloud, table_center, grasps, overall_scores = instantiate_scene_from_model(
            server,
            graspnet_ckpt,
            backend_type=scene_backend,
            config_path=config_path,
            scene_name=config_path.split('/')[-4],
            floor_height=UR5GraspPlanner.FLOOR_HEIGHT
        )
    else:
        scene_model, world_pointcloud, global_pointcloud, table_center, grasps, overall_scores = None, None, None, None, None, None

    """
    All the functions that require scene model load/train/save -- includes:
     - Load/save directory, assumes
        - data/NAME for capture dataset
        - outputs/NAME for scene model training
        - the latest written config.yml for model loading
    """
    with server.add_gui_folder("Scene load/train"):
        lerf_dataset_path = server.add_gui_text(
            label="Dataset path",
            initial_value="",
        )
        lerf_train_button = server.add_gui_button(
            label="Train/Load Scene",
        )
        lerf_reset_button = server.add_gui_button(
            label="Reset Scene",
        )
        if config_path is not None:
            lerf_dataset_path.disabled = True
            lerf_train_button.disabled = True

        @lerf_reset_button.on_click
        def _(_):
            nonlocal scene_model, world_pointcloud, global_pointcloud, table_center, grasps, overall_scores
            if scene_model is None:
                return
            scene_model.close()
            world_pointcloud, global_pointcloud, table_center, grasps, overall_scores = None, None, None, None, None
            torch.cuda.empty_cache()


        @lerf_train_button.on_click
        def _(_):
            nonlocal scene_model, world_pointcloud, global_pointcloud, table_center, grasps, overall_scores
            data = Path(f'data/{lerf_dataset_path.value}/')
            if not osp.exists(data):
                print("data file doesn't exist; can't load/train; return")
                return
            runtime_error = backend_runtime_error(scene_backend)
            if runtime_error is not None:
                print(runtime_error)
                return
            
            lerf_train_button.disabled = True
            import nerfstudio.configs.method_configs as method_configs

            num_steps = 3000
            method_name = training_method_name_for_backend(scene_backend)
            config = method_configs.all_methods[method_name]
            config.pipeline.datamanager.data = data
            config.max_num_iterations = num_steps+1
            config.steps_per_save = num_steps
            config.timestamp = "one_model"
            config.viewer.quit_on_train_completion = True

            if (
                osp.exists(config.get_base_dir()) and
                osp.exists(config.get_base_dir() / "nerfstudio_models")
            ):
                print("we are going to load a model")
                config_path = config.get_base_dir() / "config.yml"
                pipeline = None

            else:
                print("we are going to train a model")
                os.makedirs(config.get_base_dir(), exist_ok=True)
                config.save_config()

                trainer = config.setup(local_rank=0, world_size=1)
                trainer.setup()

                start = time.time()
                trainer.train()
                print(f"Training took {time.time() - start} seconds")
                pipeline = trainer.pipeline
                config_path = None
                if hasattr(pipeline.datamanager, "config") and hasattr(pipeline.datamanager.config, "patch_size"):
                    pipeline.datamanager.config.patch_size = 1
                if hasattr(pipeline.datamanager, "_get_pixel_sampler") and hasattr(pipeline.datamanager, "train_dataset"):
                    pipeline.datamanager.train_pixel_sampler = pipeline.datamanager._get_pixel_sampler(
                        pipeline.datamanager.train_dataset,
                        pipeline.datamanager.config.train_num_rays_per_batch
                    )

            scene_model, world_pointcloud, global_pointcloud, table_center, grasps, overall_scores = instantiate_scene_from_model(
                server,
                graspnet_ckpt,
                floor_height=UR5GraspPlanner.FLOOR_HEIGHT,
                backend_type=scene_backend,
                config_path=config_path,
                pipeline=pipeline,
                scene_name=lerf_dataset_path.value
            )
            lerf_train_button.disabled = False

    """
    All the functions that affect grasp generation and scores -- includes:
     - Grasp generation (for geometric scores)
     - LERF query (for semantic scores)
     - Hemisphere parameters (for generating grasps)
     - Update overall scores
     - Reset grasps

    This also includes the capability to select the best set of grasps,
    given the semantic scores and/or the geometric scores.
    """
    # Actually, prob generate the grasps when the scene is loaded.
    with server.add_gui_folder("Grasp generation"):
        with server.add_gui_folder("Semantic query"):
            gen_grasp_text = server.add_gui_text(
                label="Query (semicolon-separated)",
                initial_value=""
            )
            gen_grasp_button = server.add_gui_button(
                label="Calculate query",
            )
            primative = server.add_gui_dropdown(
                label="Primative",
                options = ["grasp",
                           "pick & place",
                           "twist",
                           "pour"],
                initial_value = "grasp"
            )
        with server.add_gui_folder("Grasp scores"):
            update_overall_scores_button = server.add_gui_button(
                label=f"Update overall scores",
            )
            update_overall_scores_threshold = server.add_gui_slider(
                label=f"Quantile threshold",
                min=0.0,
                max=1.0,
                initial_value=0.95,
                step=0.01,
            )
            update_overall_scores_slider = server.add_gui_slider(
                label=f"LERF score weight",
                min=0.0,
                max=1.0,
                initial_value=0.95,
                step=0.01,
            )

        """
        Updates LERF scores by updating the LERF query
        """
        @gen_grasp_button.on_click
        def _(_):
            nonlocal lerf_points_o3d, lerf_relevancy, grasps_dict, lerf_scores, scene_model, pick_object_pt, place_point
            nonlocal geom_scores
            gen_grasp_text.disabled = True
            gen_grasp_button.disabled = True

            lerf_weight = update_overall_scores_slider.value
            lerf_word = gen_grasp_text.value.split(";")
            print(
                "[robot_lerf] Query button clicked: "
                f"primitive={primative.value}, raw_query={gen_grasp_text.value!r}"
            )
            if primative.value == "pick & place" or primative.value == "pour":
                if len(lerf_word) != 3:
                    print(
                        "Expected query format 'object;part;place' for "
                        f"{primative.value}. Example: flower;petal;table"
                    )
                    gen_grasp_text.disabled = False
                    gen_grasp_button.disabled = False
                    return
            else:
                if len(lerf_word) != 2:
                    print(
                        "Expected query format 'object;part' for "
                        f"{primative.value}. Example: flower;petal"
                    )
                    gen_grasp_text.disabled = False
                    gen_grasp_button.disabled = False
                    return
            if scene_model is None:
                print("Load or train a scene model before running a query.")
                gen_grasp_text.disabled = False
                gen_grasp_button.disabled = False
                return

            # Get the LERF activation pointcloud for the given query
            try:
                print(f"[robot_lerf] Setting positives: {lerf_word}")
                scene_model.set_positives(lerf_word)
                print("[robot_lerf] Fetching semantic point cloud...")
                lerf_points_o3d, lerf_relevancy, dino_pcd, pick_object_pt, place_point = get_relevancy_pointcloud(scene_model, table_center=table_center)
            except NotImplementedError as exc:
                print(exc)
                gen_grasp_text.disabled = False
                gen_grasp_button.disabled = False
                return
            except Exception:
                print("[robot_lerf] Query handling failed with an unexpected exception:")
                traceback.print_exc()
                gen_grasp_text.disabled = False
                gen_grasp_button.disabled = False
                return
            # Visualize the relevancy pointcloud 
            print(
                "[robot_lerf] Semantic query returned: "
                f"semantic_pts={len(lerf_points_o3d)}, dino_pts={len(dino_pcd)}"
            )
            colors = lerf_relevancy.squeeze()
            colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-6)
            colors = matplotlib.colormaps['viridis'](colors)[:, :3]
            if lerf_weight > 0:
                semantic_display_max = int(os.environ.get("ROBOT_LERF_SEMANTIC_DISPLAY_POINTS", "60000"))
                semantic_display_points, semantic_display_colors = _subsample_points_for_display(
                    np.asarray(lerf_points_o3d),
                    colors,
                    semantic_display_max,
                )
                server.add_point_cloud(
                    name=f"/lerf_pointcloud",
                    points=semantic_display_points,
                    colors=semantic_display_colors,
                    point_size=0.007,
                )

            lerf_scores = []
            geom_scores = []
            for i, grasp in enumerate(grasps):
                box = get_bbox_from_grasp(grasp)
                # get the indices of the lerf_xyz pointcloud which lie inside the box and avg
                pts = box.get_point_indices_within_bounding_box(lerf_points_o3d)
                pts1 = box.get_point_indices_within_bounding_box(dino_pcd)
                if len(pts) == 0:
                    score = 0.0
                else:
                    # score = lerf_relevancy[pts].mean()
                    score = np.median(lerf_relevancy[pts].squeeze()).item()
                score1 = grasp.score
                if len(pts1) == 0:
                    score1 = 0.0
                
                lerf_scores.append(score)
                geom_scores.append(score1)

            #All visualization stuff
            lerf_scores = np.array(lerf_scores)
            relevancy_max = float(np.max(lerf_relevancy)) if len(lerf_relevancy) > 0 else 0.0
            if relevancy_max > 1e-6:
                lerf_scores /= relevancy_max
            else:
                print("[robot_lerf] Semantic relevancy was flat; keeping raw LERF scores.")
            print(
                "[robot_lerf] Query scoring finished: "
                f"num_grasps={len(grasps)}, num_semantic_points={len(lerf_points_o3d)}"
            )

            gen_grasp_text.disabled = False
            gen_grasp_button.disabled = False

        """
        Update the overall scores by updating the LERF score weight
        """
        @update_overall_scores_button.on_click
        def _(_):
            nonlocal grasps_dict, overall_scores

            lerf_weight = update_overall_scores_slider.value
            geom_weight = 1.0 - lerf_weight
            # Update the scores...
            if lerf_scores is None or len(lerf_scores) == 0:
                scores = geom_scores
            elif lerf_weight == 1.0:
                scores = lerf_scores
            else:
                scores = (lerf_weight)*np.array(lerf_scores) + (geom_weight)*np.array(geom_scores)
            if len(scores) == 0:
                print("Need to run LERF query first!")
                return
            scores = np.asarray(scores, dtype=np.float32)
            finite_mask = np.isfinite(scores)
            if not np.any(finite_mask):
                print("[robot_lerf] All grasp scores are non-finite; cannot rank grasps.")
                return

            scores_threshold = np.quantile(scores, update_overall_scores_threshold.value)
            inds_selected = np.flatnonzero(finite_mask & (scores >= scores_threshold))
            if inds_selected.size == 0:
                fallback_ind = int(np.nanargmax(np.where(finite_mask, scores, -np.inf)))
                inds_selected = np.array([fallback_ind], dtype=np.int64)
                print(
                    "[robot_lerf] Quantile threshold selected no grasps; "
                    f"falling back to best grasp index {fallback_ind}."
                )

            grasps_selected = [grasps[ind] for ind in inds_selected.tolist()]
            grasps_selected = GraspGroup(np.stack([grasp.grasp_array for grasp in grasps_selected]))
            # grasps_selected = grasps_selected.nms(translation_thresh=0.02, rotation_thresh=30.0/180.0*np.pi)
            grasps_selected = grasps_selected.sort_by_score()

            for ind, grasp_list in grasps_dict.items():
                for grasp in grasp_list:
                    grasp.remove()
            grasps_dict = {}

            display_scores = np.zeros_like(scores, dtype=np.float32)
            selected_scores = scores[inds_selected]
            score_min = float(selected_scores.min())
            score_max = float(selected_scores.max())
            if score_max - score_min > 1e-6:
                display_scores = (scores - score_min) / (score_max - score_min)
            display_scores = np.clip(display_scores, 0.0, 1.0)
            if score_max - score_min <= 1e-6:
                display_scores[inds_selected] = 1.0

            for ind, grasp in zip(inds_selected, grasps_selected):
                grasps_dict[int(ind)] = add_grasps(server, grasp, int(ind), float(display_scores[ind]))

            overall_scores = display_scores

    # with server.add_gui_folder("Recording"):
    #     record_path = server.add_gui_text(
    #             label="Output Path",
    #             initial_value=""
    #         )
    #     record_button = server.add_gui_button(
    #         label="Record",
    #     )
    #     cam_button = server.add_gui_button(
    #         label="Curr pos/look at",
    #     )

    #     set_pos_look_at = server.add_gui_text(
    #         label="Set pos look at",
    #         initial_value=""
    #     )

    #     cam_set_button = server.add_gui_button(
    #         label="Set",
    #     )

    #     @record_button.on_click
    #     def _(_):
    #         if record_path.value == "":
    #             print("Enter a path ")
    #             return 
            
    #         # import pdb; pdb.set_trace()
    #         sem = "sem"
    #         if update_overall_scores_slider.value == 0:
    #             sem = "geo"
    #         with server.record(Path(f"viser_assets/{record_path.value}_{sem}.viser")):
    #             time.sleep(0.1)
                
    #     @cam_button.on_click
    #     def _(_):
    #         print("Cam", curr_client.camera)



    #     @cam_set_button.on_click
    #     def _(_):
    #         if set_pos_look_at.value == "":
    #             print("Enter a path ")
    #             return 
    #         pos, look_at = set_pos_look_at.value.split(";")
    #         curr_client.camera.position = np.array([float(pos.split(",")[0]), float(pos.split(",")[1]), float(pos.split(",")[2])])
    #         curr_client.camera.look_at = np.array([float(look_at.split(",")[0]), float(look_at.split(",")[1]), float(look_at.split(",")[2])])
    while True:
        time.sleep(0.1)

if __name__ == "__main__":
    tyro.cli(main)
