# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from pytorch3d.renderer import PerspectiveCameras
from lightplane import LightplaneRenderer, Rays
from smal import SMAL
from rendering.renderer import Renderer
from model.pose_models import PoseBase
from model.pose_models.model_util import partition_index
from model.texture_models.trajectory_sampler import TrajectoryPoseDeformed


def sample_rays(masks: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample n ray indices included in the mask.

    Args:
        - mask_valid (BATCH, H, W) TORCHINT32 [DEVICE]

    Returns:
        - sampled_indices (BATCH, NPTS) TORCHINT64 [DEVICE]
        - max_indices (BATCH,) TORCHINT32 [DEVICE]
    """
    BATCH, H, W = masks.shape
    device = masks.device

    mask_flat = masks.reshape(BATCH, -1)
    sampled_indices = torch.zeros((BATCH, n), dtype=torch.int64, device=device)
    max_indices = torch.zeros((BATCH,), dtype=torch.int32, device=device)

    for b in range(BATCH):
        positive_indices = torch.nonzero(mask_flat[b], as_tuple=False).squeeze(1)
        num_positives = positive_indices.shape[0]

        if num_positives < n:
            indices_to_gather = positive_indices
        else:
            indices_to_gather = positive_indices[torch.randperm(num_positives)[:n]]

        sampled_indices[b, :num_positives] = indices_to_gather
        max_indices[b] = indices_to_gather.shape[0]

    return sampled_indices, max_indices


# Improved texture model
class TextureDuplex(nn.Module):

    def __init__(
        self,
        n_layers: int = 2,
        num_samples: int = 192,
        num_channels: int = 16,
        vol_size: int = 128,
        scale: float = 0.02,
        encode_direction: bool = False,
        gain: float = 1.0,
        opacity_init_bias: float = -5.0,
        inject_noise_sigma: float = 0.0,
        rays_jitter_near_far: bool = False,
        device: str = "cuda",
    ):
        """
        Animal Avatar duplex-mesh texture renderer.
        """
        super(TextureDuplex, self).__init__()
        self.tag = "TextureDuplex"
        self.device = device

        # Lightplane renderer
        self.renderer = LightplaneRenderer(
            num_samples=num_samples,  # Num samples along each ray
            color_chn=3,
            grid_chn=num_channels,
            mlp_hidden_chn=num_channels,
            mlp_n_layers_opacity=n_layers,
            mlp_n_layers_trunk=n_layers,
            mlp_n_layers_color=n_layers,
            enable_direction_dependent_colors=encode_direction,
            ray_embedding_num_harmonics=3 if encode_direction else None,
            gain=gain,
            opacity_init_bias=opacity_init_bias,
            inject_noise_sigma=inject_noise_sigma,
            rays_jitter_near_far=rays_jitter_near_far,
            mask_out_of_bounds_samples=False,
            bg_color=1.0,
        ).to(self.device)

        # Voxel grid
        self.xmin, self.xmax = -1.0, 2.0
        self.ymin, self.ymax = -0.5, 0.5
        self.zmin, self.zmax = -1.0, 0.5
        self.v = torch.nn.Parameter(data=torch.rand(1, vol_size, vol_size, vol_size, num_channels).contiguous(), requires_grad=True)

        # Trajectory sampler
        self.trajectory_sampler = TrajectoryPoseDeformed(scale=scale, device=self.device)

    def convert_to_voxel_space(self, X: torch.Tensor) -> torch.Tensor:
        X[..., 0] = 1 - 2 * ((self.xmax - X[..., 0]) / (self.xmax - self.xmin))
        X[..., 1] = 1 - 2 * ((self.ymax - X[..., 1]) / (self.ymax - self.ymin))
        X[..., 2] = 1 - 2 * ((self.zmax - X[..., 2]) / (self.zmax - self.zmin))
        return X

    def render(self, batch_points_in_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Raymarch between IN and OUT points.

        Args:
            - points_in_out (BATCH, 2, 3) TORCHFLOAT32 IN NDC

        Returns:
            - rendered_features: (BATCH, 3) TORCHFLOAT32
            - rendered_mask: (BATCH, 1) TORCHFLOAT32
        """
        N_RAYS, _, _ = batch_points_in_out.shape
        device = batch_points_in_out.device

        # Ray directions ->(BATCH, 3)
        directions = batch_points_in_out[:, 1] - batch_points_in_out[:, 0]
        # Ray centers ->(BATCH, 3)
        origins = batch_points_in_out[:, 0]
        # Ray near/far plane ->(BATCH,)
        near = torch.zeros((N_RAYS,), dtype=torch.float32, device=device)
        far = torch.ones((N_RAYS,), dtype=torch.float32, device=device)
        # Ray grid_idx ->(BATCH,)
        grid_idx = torch.zeros((N_RAYS,), dtype=torch.int32, device=device)

        # Create rendering Rays
        rays = Rays(
            directions=directions,
            origins=origins,
            grid_idx=grid_idx,
            near=near,
            far=far,
        )

        rendered_ray_length, rendered_alpha, rendered_features = self.renderer(rays=rays, feature_grid=[self.v])

        return rendered_features, rendered_alpha[:, None]

    def render_partition(self, batch_points_in_out: torch.Tensor, chunksize: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            - batch_points_in_out (BATCH, 2, 3) TORCHFLOAT32 IN NDC

        Returns:
            - rendered_features: (BATCH, 3) TORCHFLOAT32
            - rendered_mask: (BATCH, 1) TORCHFLOAT32

        """
        BATCH, _, _ = batch_points_in_out.shape
        device = batch_points_in_out.device

        indexes_chunk = partition_index(totalsize=batch_points_in_out.shape[0], chunksize=chunksize)

        # Create output features and masks
        rendered_features = torch.zeros((BATCH, 3), dtype=torch.float32, device=device).contiguous()
        rendered_masks = torch.zeros((BATCH, 1), dtype=torch.float32, device=device).contiguous()

        # We need to keep the same chunksize, even at the end, so we compute larger batch and keep only a subset
        for a, b in indexes_chunk:
            if (b - a) != chunksize:
                c = b - a
                input_render = torch.zeros((chunksize, 2, 3), dtype=torch.float32, device=device)
                input_render[:c] = batch_points_in_out[a:b]
                chunk_rend_feat, chunk_rend_mask = self.render(input_render)
                rendered_features[a:b] = chunk_rend_feat[:c]
                rendered_masks[a:b] = chunk_rend_mask[:c]
            else:
                chunk_rend_feat, chunk_rend_mask = self.render(batch_points_in_out[a:b])
                rendered_features[a:b] = chunk_rend_feat
                rendered_masks[a:b] = chunk_rend_mask

        return rendered_features, rendered_masks

    def render_image_sampling(
        self,
        n_rays_per_image: int,
        smal: SMAL,
        renderer: Renderer,
        pose_model: PoseBase,
        X_ind: torch.Tensor,
        X_ts: torch.Tensor,
        cameras: PerspectiveCameras,
        vertices: torch.Tensor | None = None,
        faces: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Render a subset of image pixels.

        Returns:
            - rendered_ray_images (BATCH, N_RAYS_PER_IMAGE, 3) TORCHFLOAT32
            - sampled_indices (BATCH, N_RAY_PER_IMAGE) TORCHINT32
            - max_indice (BATCH,) TORCHINT32
        """
        batch_points_in_out_ndc, mask_valid = self.trajectory_sampler.generate_point_map_in_out(renderer, pose_model, X_ind, X_ts, cameras, vertices, faces)
        batch_points_in_out_ndc = batch_points_in_out_ndc.reshape(BATCH, H * W, 2, 3)
        BATCH, H, W, _, _ = batch_points_in_out_ndc.shape

        # Sample rays from images ->(BATCH, N_RAYS_PER_IMAGE, 2, 3)
        sampled_indices, max_indices = sample_rays(mask_valid, n_rays_per_image)
        X_input_ndc = torch.gather(batch_points_in_out_ndc, index=sampled_indices.reshape(BATCH, n_rays_per_image, 1, 1).expand(-1, -1, 2, 3), dim=1)

        # Convert to voxel NCC space [-1,1]x[-1,1]x[-1,1]
        X_input_ndc = self.convert_to_voxel_space(X_input_ndc)

        rendered_ray_images, _ = self.render_partition(X_input_ndc.reshape(-1, 2, 3), chunksize=1024)
        rendered_ray_images = rendered_ray_images.reshape((BATCH, n_rays_per_image, 3))

        return rendered_ray_images, sampled_indices, max_indices

    def render_images(
        self,
        smal: SMAL,
        renderer: Renderer,
        pose_model: PoseBase,
        X_ind: torch.Tensor,
        X_ts: torch.Tensor,
        cameras: PerspectiveCameras,
        vertices: torch.Tensor | None = None,
        faces: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render the full images.

        Returns:
            - rendered_images (BATCH, H, W, 3) [0,1] TORCHFLOAT32 [DEVICE]
            - rendered_masks (BATCH, H, W, 1) {0,1} TORCHINT32 [DEVICE]
        """
        batch_points_in_out_ndc, mask_valid = self.trajectory_sampler.generate_point_map_in_out(renderer, pose_model, X_ind, X_ts, cameras, vertices, faces)
        BATCH, H, W, _, _ = batch_points_in_out_ndc.shape
        device = batch_points_in_out_ndc.device

        # Process all rays in mask_valid
        valid_indices = (
            mask_valid.reshape(
                -1,
            )
            .nonzero()
            .reshape(
                -1,
            )
        )
        batch_points_in_out_ndc_to_process = batch_points_in_out_ndc.reshape(BATCH * H * W, 2, 3)[valid_indices]

        # Convert to voxel NDC space [-1,1]x[-1,1]x[-1,1]
        batch_points_in_out_ndc_to_process = self.convert_to_voxel_space(batch_points_in_out_ndc_to_process)

        # Raymarch
        renders_img_rgb, renders_mask = self.render_partition(batch_points_in_out_ndc_to_process, chunksize=2**16)

        # 1) Create output image tensor
        rendered_images = torch.zeros((BATCH * H * W, 3), dtype=torch.float32, device=device)
        rendered_images[valid_indices] = renders_img_rgb
        rendered_images = rendered_images.reshape((BATCH, H, W, 3))

        # 2) Create output mask tensor
        rendered_masks = torch.zeros((BATCH * H * W, 1), dtype=torch.float32, device=device)
        rendered_masks[valid_indices] = renders_mask
        rendered_masks = rendered_masks.reshape((BATCH, H, W, 1))

        return rendered_images, rendered_masks
