# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.camera_utils import join_cameras_as_batch
import utils.img_utils as imutil
from utils.img_utils import resize_image


class CustomSingleVideo(Dataset):
    """
    An example custom dataset class
    To train Animal-Avatar on a custom scene, complete this Dataset (except get_sparse_keypoints()) with your data
    """

    def __init__(self, *args, **kwargs):
        self.rgb_img_path = "/workspaces/AnimalAvatar/cop3d_data/dog/565_81664_160332/images"
        self.mask_path = "external_data/refined_masks/565_81664_160332"
        self.camera_path = "/workspaces/AnimalAvatar/cop3d_data/dog/565_81664_160332/cameras/cameras.pt" # This is a List object containing PerspectiveCameras for each frame
        
        self.sequence_frame_ids = []
        
        for filename in os.listdir(self.rgb_img_path):
            if filename.endswith(".jpg"):
                self.sequence_frame_ids.append(filename.split(".")[0]) # removes filetype extension
                
        self.sequence_frame_ids.sort()
        print(f"Found {len(self.sequence_frame_ids)} frames in the sequence. Some of it are: {self.sequence_frame_ids[:5]}")
                
        # check if all frames have corresponding masks
        actual_mask_files = set(os.listdir(self.mask_path))
        expected_mask_files = {f"{frame_id}.png" for frame_id in self.sequence_frame_ids}
        assert len(expected_mask_files - actual_mask_files) == 0, "Missing mask files."
        assert len(actual_mask_files - expected_mask_files) == 0, "Extra mask files."
        
        # TODO add check for cameras
        

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
            X = np.array(Image.open(mask_path)).astype(np.uint8)
            X = np.expand_dims(X, -1) # add C dimension, is now (H,W,1)

            # TODO cop3d dataloader always rescales to 800x800. This might be important for https://pytorch3d.readthedocs.io/en/latest/modules/renderer/cameras.html#pytorch3d.renderer.cameras.CamerasBase -> NDC coordinate system, but I didnt check this yet NOTE this needs TRANSPOSING as done in cop3d_dataloader.py
            
            mask_list.append(X.astype(np.float32))
            
        return np.array(mask_list, dtype=np.float32) # has shape (202, 483, 360, 1)
        
    def get_imgs_rgb(self, list_items: list[int]) -> np.ndarray:
        """            
        Return:
            - imgs_rgb [len(list_items),H,W,3] NPFLOAT32 [0,1] (rgb frames)
        """
        imgs_rgb = []
        for idx in list_items:
            img_path = os.path.join(self.rgb_img_path, self.sequence_frame_ids[idx] + ".jpg")
            img = Image.open(img_path).convert("RGB")  # Ensure it's in RGB

            img = np.true_divide(img, 255, dtype=np.float32)
            
            # TODO cop3d dataloader always rescales to 800x800. This might be important for https://pytorch3d.readthedocs.io/en/latest/modules/renderer/cameras.html#pytorch3d.renderer.cameras.CamerasBase -> NDC coordinate system, but I didnt check this yet NOTE this might need TRANSPOSING as done in cop3d_dataloader.py
            
            
            imgs_rgb.append(img)

        return np.array(imgs_rgb, dtype=np.float32)

    def get_cameras(self, list_items: list[int]) -> PerspectiveCameras:
        """
        Return:
            - cameras (len(list_items),) PerspectiveCameras in ndc_space
        """
        cameras = torch.load(self.camera_path)
        return join_cameras_as_batch([cameras[i] for i in list_items])

    def get_sparse_keypoints(self, list_items: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """
        Optional!
        Returns:
            - sparse_keypoints (N, N_KPTS, 2) NPFLOAT32
            - scores (N, N_KPTS, 1) NPFLOAT32
        """
        assert False, ("No GT sparse-keypoints are available for custom scenes.", "Please set exp.l_optim_sparse_kp=0 in config/config.yaml", "This will affect performance quality.")
