# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    BlendParams,
    PointLights,
    HardPhongShader,
    SoftSilhouetteShader,
    TexturesVertex,
)


def partition_index(totalsize: int, chunksize: int) -> list[tuple[int, int]]:
    a, b = 0, 0
    if totalsize <= chunksize:
        return [(a, totalsize)]
    list_tuples = []
    b = chunksize
    while b < totalsize:
        list_tuples.append((a, b))
        a = b
        b += chunksize
    list_tuples.append((a, min(totalsize, b)))
    return list_tuples


class Renderer(nn.Module):
    def __init__(self, image_size: int, black_bg: bool = True, faces_per_pixels: int = 100):
        """
        Animal Avatar Pytorch3d-based renderer.
        """
        super(Renderer, self).__init__()

        self.image_size = image_size
        self.color_bg = (1, 1, 1) if black_bg else (255, 255, 255)
        self.blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=self.color_bg)

        self.raster_settings_soft = RasterizationSettings(image_size=image_size, blur_radius=np.log(1.0 / 1e-4 - 1.0) * self.blend_params.sigma, faces_per_pixel=faces_per_pixels)
        self.raster_settings_vis = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)

    def get_classical_texture(self, vertices: torch.Tensor) -> torch.Tensor:
        mesh_color = torch.tensor([0, 172, 223], dtype=torch.float32, device=vertices.device)
        return torch.ones_like(vertices) * mesh_color

    def render_visualization(self, vertices: torch.Tensor, faces: torch.Tensor, cameras: PerspectiveCameras, lights=None, texture: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            - vertices (BATCH, 3889, 3) TORCHFLOAT32
            - faces (BATCH, 7773, 3) TORCHINT64
            - cameras (PerpectiveCameras BATCH)

        Return:
            - visualization (BATCH, 3, IMG_SIZE0, IMG_SIZE1) TORCHFLOAT32 [0,255]
        """
        device = vertices.device

        if texture is None:
            texture = self.get_classical_texture(vertices)

        if lights is None:
            lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

        mesh = Meshes(verts=vertices, faces=faces, textures=TexturesVertex(verts_features=texture))

        vis_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings_vis), shader=HardPhongShader(device=device, cameras=cameras, lights=lights, blend_params=self.blend_params)
        )

        visualization = vis_renderer(mesh).permute(0, 3, 1, 2)[:, :3, :, :]
        return visualization

    def render_silhouette(self, vertices: torch.Tensor, faces: torch.Tensor, cameras: PerspectiveCameras) -> torch.Tensor:
        """
        Args:
            - vertices (BATCH, N_VERTS, 3) TORCHFLOAT32
            - faces (BATCH, N_FACES, 3) TORCHINT64
            - cameras (BATCH,)
        Return:
            - silh_images (BATCH, 1, H_rend, W_rend) TORCHFLOAT32 [DEVICE] [0,1]
        """

        mesh = Meshes(verts=vertices, faces=faces)

        # silhouette renderer
        renderer_silh = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings_soft), shader=SoftSilhouetteShader(blend_params=self.blend_params))

        silh_images = renderer_silh(mesh)[..., -1].unsqueeze(1)
        return silh_images

    def render_silhouette_partition(self, vertices: torch.Tensor, faces: torch.Tensor, cameras: PerspectiveCameras, chunksize: int = 100) -> torch.Tensor:
        """
        Args:
            - vertices (BATCH, N_VERTS, 3) TORCHFLOAT32
            - faces (BATCH, N_FACES, 3) TORCHINT64
            - cameras (BATCH,)
        Return:
            - silh_images (BATCH, 1, H_nvs, W_nvs) TORCHFLOAT32 [0,1]
        """
        indexes_chunk = partition_index(totalsize=vertices.shape[0], chunksize=chunksize)
        return torch.cat([self.render_silhouette(vertices[a:b], faces[a:b], cameras[list(range(a, b))]) for (a, b) in indexes_chunk], dim=0)

    def render_cse(self, vertices: torch.Tensor, faces: torch.Tensor, cameras: PerspectiveCameras, cse_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render cse_embedding of meshes.

        Args:
            - vertices (BATCH, N_v, 3)
            - faces (BATCH, N_f, 3)
            - cameras: PerspectiveCamera Batch
            - cse_embedding (N_v, 16)

        Return
            - rendered_cse (BATCH, IMG_W, IMG_H, 16) : rendered CSE embedding for all the batch
            - mask (BATCH, IMG_W, IMG_H, 1) TORCHBOOL
        """

        mesh = Meshes(verts=vertices, faces=faces)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings_vis)
        fragments = rasterizer(mesh)

        BATCH = len(mesh.verts_list())
        faces_packed = mesh.faces_packed()
        cse_embeddings_packed = cse_embedding.expand(BATCH, -1)

        rendered_cse = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, cse_embeddings_packed[faces_packed])
        rendered_mask = fragments.pix_to_face != -1

        return rendered_cse, rendered_mask

    def render_render_cse_partition(
        self, vertices: torch.Tensor, faces: torch.Tensor, cameras: PerspectiveCameras, cse_embedding: torch.Tensor, chunksize: int = 20, device: str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            - vertices (BATCH, N_v, 3)
            - faces (BATCH, N_f, 3)
            - cameras: PerspectiveCamera (Batch,)
            - cse_embedding (N_v, 16)
        Return
            - rendered_cse (BATCH, IMG_W, IMG_H, 16) TORCHFLOAT32
            - rendered_mask (BATCH, IMG_W, IMG_H, 1) TORCHBOOL
        """

        indexes_chunk = partition_index(totalsize=vertices.shape[0], chunksize=chunksize)

        rendered_cse = torch.zeros((vertices.shape[0], self.image_size, self.image_size, 1, cse_embedding.shape[-1]), dtype=torch.float32, device=device)
        rendered_mask = torch.zeros((vertices.shape[0], self.image_size, self.image_size, 1), dtype=torch.bool, device=device)

        for a, b in indexes_chunk:
            output = self.render_cse(vertices[a:b].contiguous(), faces[a:b].contiguous(), cameras[list(range(a, b))], cse_embedding.contiguous())
            rendered_cse[a:b] = output[0].to(device)
            rendered_mask[a:b] = output[1].to(device)

        return rendered_cse, rendered_mask

    def get_visibility_map(self, vertices: torch.Tensor, faces: torch.Tensor, cameras: PerspectiveCameras) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            - vertices (BATCH, N_vertices,3)
            - faces (BATCH, N_faces, 3)
            - cameras (BATCH) PerspectiveCameras

        Return:
            - vertex_visibility_map (BATCH, N_vertices) TORCHINT32 {0,1}
            - faces_visibility_map (BATCH, N_faces) TORCHINT32 {0,1}
        """

        BATCH, N_verts, _ = vertices.shape
        _, N_faces, _ = faces.shape
        device = vertices.device

        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings_vis)

        meshes = Meshes(verts=vertices, faces=faces)

        # Get the output from rasterization
        fragments = rasterizer(meshes)

        # pix_to_face is of shape (N, H, W, 1)
        pix_to_face = fragments.pix_to_face

        # (F, 3) where F is the total number of faces across all the meshes in the batch
        packed_faces = meshes.faces_packed()
        # (V, 3) where V is the total number of verts across all the meshes in the batch
        packed_verts = meshes.verts_packed()

        # Indices of unique visible faces
        visible_faces = pix_to_face.unique()  # (num_visible_faces,)
        visible_faces = visible_faces[visible_faces > 0]

        # Get Indices of unique visible verts using the vertex indices in the faces
        visible_verts_idx = packed_faces[visible_faces]  # (num_visible_faces,  3)
        unique_visible_verts_idx = torch.unique(visible_verts_idx)  # (num_visible_verts, )

        # Output tensors
        vertex_is_visible = torch.zeros(packed_verts.shape[0], dtype=torch.int32, device=device)
        face_is_visible = torch.zeros(packed_faces.shape[0], dtype=torch.int32, device=device)

        # Update visibility indicator to 1 for all visible vertices
        vertex_is_visible[unique_visible_verts_idx] = 1
        vertex_is_visible = vertex_is_visible.reshape((BATCH, N_verts))

        face_is_visible[visible_faces] = 1
        face_is_visible = face_is_visible.reshape((BATCH, N_faces))

        return vertex_is_visible, face_is_visible

    def get_rasterization(self, vertices: torch.Tensor, faces: torch.Tensor, cameras: PerspectiveCameras) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get rasterization results with a single face per pixel.
        Returns:
            - pix_to_face (BATCH, H_rend, W_rend, 1) TORCHLONG
            - baryc_coord (BATCH, H_rend, W_rend, 1, 3) TORCHFLOAT32
            - valid_mask: (BATCH, H_rend, W_rend) TORCHBOOL
        """
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings_vis)

        meshes = Meshes(vertices, faces)
        fragments = rasterizer(meshes)

        faces_hit = fragments.pix_to_face
        baryc_coord = fragments.bary_coords
        valid_mask = fragments.pix_to_face[..., 0] != -1

        return faces_hit, baryc_coord, valid_mask
