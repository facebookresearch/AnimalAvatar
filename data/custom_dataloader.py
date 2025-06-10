# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import yaml
import numpy as np
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from pytorch3d.transforms import quaternion_to_matrix
import utils.img_utils as imutil
from utils.img_utils import resize_image
from data.utils import array_list_to_stack


class CustomSingleVideo(Dataset):
    """
    An example custom dataset class
    To train Animal-Avatar on a custom scene, complete this Dataset (except get_sparse_keypoints()) with your data
    """

    def __init__(
        self,
        device,
        cop3d_root_path,
        sequence_index,
        *args,
        **kwargs,
    ):
        self.device = device
        self.cop3d_root_path = cop3d_root_path
        self.sequence_index = sequence_index
        self.subfolder = "maila"

        self.image_width_height = 800 # those are the resize values used by COP3D. Improves performance. See README.md

        path = os.path.join(self.cop3d_root_path, self.subfolder,self.sequence_index)
        self.rgb_img_path = os.path.join(path, "rgb")
        self.mask_path =  os.path.join(path, "masks")
        self.camera_path = f"{path}/traj_est.npy" # This is a List object containing PerspectiveCameras for each frame

        self.sequence_frame_ids = []
        self.img_file_suffix = None

        for filename in os.listdir(self.rgb_img_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                self.sequence_frame_ids.append(filename.split(".")[0]) # removes filetype extension
                if self.img_file_suffix == None:
                    self.img_file_suffix = filename.split(".")[1]
                else:
                    # we only expect all image files to be of same filetype
                    assert self.img_file_suffix == filename.split(".")[1]

        self.sequence_frame_ids.sort()

        # check if all frames have corresponding masks
        actual_mask_files = set(os.listdir(self.mask_path))
        expected_mask_files = {f"{frame_id}.png" for frame_id in self.sequence_frame_ids}
        assert len(expected_mask_files - actual_mask_files) == 0, "Missing mask files."
        assert len(actual_mask_files - expected_mask_files) == 0, "Extra mask files."

        # camera
        self.traj_est = torch.tensor(np.load(self.camera_path))
        with open("cop3d_data/maila/camera.yaml", "r") as file:
            camera_info = yaml.safe_load(file)
        self.FX, self.FY, self.CX, self.CY = camera_info["intrinsics"]

        # check if all frames have camera
        assert len(self.traj_est) == len(actual_mask_files)

        print(f"Found {len(self.sequence_frame_ids)} frames in the sequence. Some of it are: {self.sequence_frame_ids[:3]}")
        
        # for cop3d_data/dog/565_81664_160332 you can just use the following
        # Additionally, remove most of the logic above
        # self.rgb_img_path = "cop3d_data/dog/565_81664_160332/images"
        # self.mask_path = "external_data/refined_masks/565_81664_160332"
        # self.camera_path = "cop3d_data/dog/565_81664_160332/cameras/cameras.pt"

    def __len__(self):
        return len(self.sequence_frame_ids)

    def get_masks(self, list_items: list[int]) -> np.ndarray:
        """
        
        Return:
            - masks [len(list_items),H,W,1] NPFLOAT32 [0,1](foreground probability)
        """

        # TODO cop3d dataloader always rescales to 800x800. This might be important for https://pytorch3d.readthedocs.io/en/latest/modules/renderer/cameras.html#pytorch3d.renderer.cameras.CamerasBase -> NDC coordinate system, but I didnt check this yet

        mask_list = []
        for idx in list_items:
            mask_path = os.path.join(self.mask_path, self.sequence_frame_ids[idx] + ".png")
            X = np.array(Image.open(mask_path)).astype(np.uint8) # HxW
            X = np.expand_dims(X, 0) # add C dimension, is now (1,H,W,)
            X_resized = resize_image(X, self.image_width_height, self.image_width_height, mode="bilinear")[0].unsqueeze(0).numpy()

            mask_list.append(np.transpose(X_resized.astype(np.float32), (0, 2, 3, 1))) # reordering dimensions to (1,self.image_width_height,self.image_width_height,1)

        return array_list_to_stack(mask_list) # has shape (202, 483, 360, 1)

    def get_imgs_rgb(self, list_items: list[int]) -> np.ndarray:
        """            
        Return:
            - imgs_rgb [len(list_items),H,W,3] NPFLOAT32 [0,1] (rgb frames)
        """
        imgs_rgb = []
        for idx in list_items:
            img_path = os.path.join(self.rgb_img_path, self.sequence_frame_ids[idx] + "." + self.img_file_suffix )
            img = Image.open(img_path)
            img = np.true_divide(img, 255, dtype=np.float32) # (HxWx3)

            img_resized = resize_image(np.transpose(img,(2,0,1)), self.image_width_height, self.image_width_height, mode="bilinear")[0].numpy()

            img_resized = np.transpose(img_resized.astype(np.float32), (1,2,0)) # (HxWx3)

            imgs_rgb.append(img_resized)

        return np.array(imgs_rgb, dtype=np.float32) # has to be (N,H,W,3)

    def get_cameras(self, list_items: list[int]) -> PerspectiveCameras:
        """
        Return:
            - cameras (len(list_items),) PerspectiveCameras in ndc_space
        """

        T = self.traj_est[list_items, :3].to(self.device)
        R_quat = self.traj_est[list_items, 3:].to(self.device)

        R = quaternion_to_matrix(R_quat)

        fx_ndc, fy_ndc, px_ndc, py_ndc = (
            2.0 * self.FX / self.image_width_height,
            2.0 * self.FY / self.image_width_height,
            -(2.0 * (self.CX / self.image_width_height) - 1.0),
            -(2.0 * (self.CY / self.image_width_height) - 1.0),
        )

        N = len(list_items)
        focal_length = torch.tensor([[fx_ndc, fy_ndc]] * N, device=self.device)  # (N, 2)
        principal_point = torch.tensor([[px_ndc, py_ndc]] * N, device=self.device)  # (N, 2)
        image_size = torch.tensor(
            [[self.image_width_height, self.image_width_height]] * N, device=self.device
        )  # (N, 2)

        cameras = PerspectiveCameras(
            R=R,
            T=T,
            focal_length=focal_length,
            principal_point=principal_point,
            in_ndc=True,
            image_size=None, #image_size,
            device=self.device,
        )

        return cameras

        # for cop3d_data/dog/565_81664_160332 you can just use the following:
        # cameras = torch.load(self.camera_path)
        # return join_cameras_as_batch([cameras[i] for i in list_items])

    def get_sparse_keypoints(self, list_items: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """
        Optional!
        Returns:
            - sparse_keypoints (N, N_KPTS, 2) NPFLOAT32
            - scores (N, N_KPTS, 1) NPFLOAT32
        """
        assert False, ("No GT sparse-keypoints are available for custom scenes.", "Please set exp.l_optim_sparse_kp=0 in config/config.yaml", "This will affect performance quality.")
