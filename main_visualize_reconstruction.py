# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from data.input_cop import get_input_cop_from_archive
from model.inferencer import Inferencer
import rendering.visualization as viz
import rendering.visualization_renderer as vizrend
from scene_optim.callbacks import CallbackEval
from pytorch3d.structures import Meshes
from pytorch3d.structures import Pointclouds

from model.pose_models import compute_pose
from pytorch3d.io import save_obj, save_ply

def rotate_robot_upright_vectorized(base, feet):
    """
    Vectorized version for better performance with large datasets.
    
    Parameters and returns are the same as rotate_robot_upright().
    """
    
    def objective_function(rotation_params):
        """Vectorized objective function"""
        rotation = R.from_rotvec(rotation_params)
        rot_matrix = rotation.as_matrix()
        
        # Calculate centroids across all frames at once
        base_centroids = np.mean(base, axis=0)  # (frames, 3)
        feet_centroids = np.mean(feet, axis=0)  # (frames, 3)
        
        # Rotate centroids (vectorized)
        rotated_base_centroids = (rot_matrix @ base_centroids.T).T
        rotated_feet_centroids = (rot_matrix @ feet_centroids.T).T
        
        # Calculate total z-difference across all frames
        z_diffs = rotated_base_centroids[:, 2] - rotated_feet_centroids[:, 2]
        total_z_diff = np.sum(z_diffs)
        
        return -total_z_diff
    
    # Optimize rotation
    initial_rotation = np.array([0.0, 0.0, 0.0])
    result = minimize(objective_function, initial_rotation, method='BFGS')
    
    # Apply optimal rotation
    optimal_rotation = R.from_rotvec(result.x)
    optimal_rot_matrix = optimal_rotation.as_matrix()
    
    # Vectorized rotation application
    base_reshaped = base.reshape(-1, 3)  # (markers*frames, 3)
    feet_reshaped = feet.reshape(-1, 3)  # (markers*frames, 3)
    
    rotated_base_reshaped = (optimal_rot_matrix @ base_reshaped.T).T
    rotated_feet_reshaped = (optimal_rot_matrix @ feet_reshaped.T).T
    
    rotated_base = rotated_base_reshaped.reshape(base.shape)
    rotated_feet = rotated_feet_reshaped.reshape(feet.shape)
    
    return rotated_base, rotated_feet

def save_skeleton_with_edges_to_obj_pytorch3d(points, edges, filename="skeleton_with_edges.obj", filter_point_of_interest=False):
    """
    Save 3D skeleton points with edges as an OBJ file using PyTorch3D.
    
    points: (N, 3) numpy array of 3D coordinates
    edges: List of (i, j) tuples defining connections (as line segments)
    filename: Output OBJ filename
    """
    # If filter_point_of_interest is True, select only the 0th, 3rd, 6th, 9th keypoints
    if filter_point_of_interest:
        points = points[[0, 3, 6, 9, 12, 22]]  # first four 4 feet, second two are base
    
    # Convert points to PyTorch tensor
    verts = torch.tensor(points, dtype=torch.float32)

    # Prepare the OBJ file content
    obj_content = []

    # Write vertices (v x y z)
    for point in points:
        obj_content.append(f"v {point[0]} {point[1]} {point[2]}")

    # Write edges (l i j) as lines
    if not filter_point_of_interest:
        for (i, j) in edges:
            obj_content.append(f"l {i+1} {j+1}")  # OBJ indexing starts from 1, not 0

    # Write to the OBJ file
    with open(filename, "w") as f:
        f.write("\n".join(obj_content))

    print(f"Saved {filename}")
    
def visualize_reconstruction(archive_path: str, device: str = "cuda"):

    # Load inputs
    ic = get_input_cop_from_archive(archive_path, device=device)

    image_size = ic.image_size
    images = ic.images.to(device)
    masks = ic.masks.to(device)
    X_ind = ic.X_ind
    X_ts = ic.X_ts
    X_ind_test = ic.X_ind_test
    X_ts_test = ic.X_ts_test
    smal = ic.smal
    renderer = ic.renderer
    texture = ic.texture.to(device)
    cameras = ic.cameras.to(device)
    cse_embedding = ic.cse_embedding.to(device)

    # Load model (pose, texture)
    inferencer = Inferencer(archive_path, use_archived_code=True)
    pose_model = inferencer.load_pose_model().to(device)
    texture_model = inferencer.load_texture_model().to(device)

    with torch.no_grad():
        vertices, keypoints_3d, faces = compute_pose(smal, pose_model, X_ind, X_ts)
        
    mesh_save_path = os.path.join(archive_path, "meshes")
    keypoints_save_path = os.path.join(archive_path, "keypoints")
    os.makedirs(mesh_save_path, exist_ok=True)
    os.makedirs(keypoints_save_path, exist_ok=True)
    os.makedirs(os.path.join(mesh_save_path, "pointclouds"), exist_ok=True)
    os.makedirs(os.path.join(mesh_save_path, "meshes"), exist_ok=True)

    for i in range(len(vertices)):
        # save mesh
        mesh = Meshes(verts=vertices[i].unsqueeze(0), faces=faces[i].unsqueeze(0))
        save_obj(os.path.join(mesh_save_path, "meshes", f"test_{i}.obj"), mesh.verts_packed(), mesh.faces_packed())
        
        # save pointcloud
        pointcloud = Pointclouds(points=vertices[i].unsqueeze(0))
        save_ply(os.path.join(mesh_save_path, "pointclouds", f"test_{i}.ply"), pointcloud.points_packed())
    edges = [(i, i+1) for i in range(23)]  
    for i in range(len(keypoints_3d)):
        save_skeleton_with_edges_to_obj_pytorch3d(keypoints_3d[i], edges, os.path.join(keypoints_save_path, f"keypoints_{i}.obj"))
        save_skeleton_with_edges_to_obj_pytorch3d(keypoints_3d[i], edges, os.path.join(keypoints_save_path, f"keypoints_tracks_{i}.obj"), filter_point_of_interest=True)
        
    # save as required by motion retargeting
    base = keypoints_3d[:,[12,22],:].cpu().numpy().transpose(1, 0, 2)
    feet = keypoints_3d[:,[0,3,6,9],:].cpu().numpy().transpose(1, 0, 2)
    
    base, feet = rotate_robot_upright_vectorized(base, feet)
    
    np.save(f"{archive_path}/base.npy", base)
    np.save(f"{archive_path}/feet.npy", feet)
    print("Saved base and feet arrays as numpy.")
    

    # -- STATS
    callback_eval = CallbackEval(images, masks, cameras, smal, cse_embedding, image_size, device)

    final_eval, final_eval_str = callback_eval.call(pose_model, X_ind_test, X_ts_test, texture_model)
    print("Results: {}".format(final_eval_str))

    # -- RENDERING
    viz.make_video_list(
        vizrend.global_visualization(images, masks, pose_model, X_ind, X_ts, smal, texture_model, cse_embedding, renderer, cameras, texture=texture),
        os.path.join(archive_path, "rendered_optimized.mp4"),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize reconstruction results")
    parser.add_argument("archive_path", type=str, help="Archive of the reconstruction")
    args = parser.parse_args()

    print(f"Archive path: {args.archive_path}")

    visualize_reconstruction(args.archive_path)
