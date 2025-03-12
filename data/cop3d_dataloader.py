# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import pickle as pk
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from pytorch3d.implicitron.dataset.sql_dataset import SqlIndexDataset
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from utils.img_utils import resize_image
from config.keys import Keys
from data.utils import array_list_to_stack


def crop_to_mask(mask) -> tuple[int, int, int, int]:
    mask_nz = mask.nonzero()
    x0, x1 = int(mask_nz[:, 2].min().item()), int(mask_nz[:, 2].max().item())
    y0, y1 = int(mask_nz[:, 1].min().item()), int(mask_nz[:, 1].max().item())
    w = x1 - x0
    h = y1 - y0
    return x0, y0, w, h


def _is_valid_sequence_name(cop3d_dataset: SqlIndexDataset, sequence_index: str) -> bool:
    return sequence_index in cop3d_dataset.sequence_names()


def load_cop3d_sql_dataset(dataset_root: str, box_crop: bool = False, resizing: bool = True) -> SqlIndexDataset:

    metadata_file = os.path.join(dataset_root, "metadata.sqlite")
    assert os.path.isfile(metadata_file)

    # Load the COP3D dataloader
    dataset = SqlIndexDataset(sqlite_metadata_file=metadata_file, dataset_root=dataset_root)
    dataset.frame_data_builder.load_depths = False  # the dataset does not provide depth maps

    # To turn off cropresizing
    if box_crop:
        dataset.frame_data_builder.box_crop = True
    else:
        dataset.frame_data_builder.box_crop = False
    if resizing == False:
        dataset.frame_data_builder.image_height = None  # turns off resizing

    return dataset


def cameras_from_metadatas(cameras: PerspectiveCameras, device: str, original_cameras: bool = False) -> PerspectiveCameras:

    new_cameras = cameras.clone()

    # Original intrinsics and extrinsics
    if original_cameras:
        return new_cameras.to(device)

    # Original intrinsics but fixed extrinsics
    else:
        N = new_cameras.R.shape[0]
        new_cameras.R = torch.Tensor(np.eye(3)).float()[None, :, :].to(device).repeat((N, 1, 1))
        new_cameras.T = torch.zeros((N, 3))
        return new_cameras.to(device)


class COPSingleVideo(Dataset):

    def __init__(self, cop3d_root_path: str, sequence_index: str, cop3d_cropping: bool = False, cop3d_resizing: bool = True, preload: bool = True):
        """
        COP3D Dataset for a single video indexed by 'sequence_index'.
        Args:
            - cop3d_cropping: If True, crop the image around the mask. Affect the cameras (principal_point, focal_length)
            - cop3d_resizing: If True, square-pad the image and down-resize to (800,800)
        """
        self.cop3d_root_path = cop3d_root_path
        self.sequence_index = sequence_index
        self.cop3d_cropping = cop3d_cropping
        self.cop3d_resizing = cop3d_resizing
        self.preload = preload

        # Load the SQL index cop3d dataset
        self.cop3d_dataset = load_cop3d_sql_dataset(self.cop3d_root_path, box_crop=cop3d_cropping, resizing=self.cop3d_resizing)
        assert _is_valid_sequence_name(self.cop3d_dataset, self.sequence_index), f"Invalid sequence_index: {self.sequence_index}"

        # Preload the frames
        self.sequence_frame_ids = [frame.frame_number for frame in self.cop3d_dataset.sequence_frames_in_order(self.sequence_index)] # list of frame_ids, eg [1,2,3 ...]

        if self.preload:
            self.sequence_frames_dict = {id_: self.cop3d_dataset[self.sequence_index, id_] for id_ in tqdm(self.sequence_frame_ids, desc="Loading COP3D frames ({})".format(self.sequence_index))}

    def __len__(self):
        return len(self.sequence_frame_ids)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Return:
            - cropped frame (1x3x800x800) TORCHFLOAT32 [0,1]
            - cropped mask (1x1x800x800) TORCHFLOAT32 [0,1]
            - camera metadata
        """
        if self.preload:
            frame_i = self.sequence_frames_dict[self.sequence_frame_ids[i]]
        else:
            frame_i = self.cop3d_dataset[self.sequence_index, self.sequence_frame_ids[i]]

        img_ = frame_i.image_rgb.unsqueeze(0)
        mask_ = frame_i.fg_probability.unsqueeze(0)

        mask_crop = frame_i.mask_crop
        if mask_crop is not None:
            crop_bbox = crop_to_mask(mask_crop)
        else:
            crop_bbox = [0, 0, int(img_.shape[2]), int(img_.shape[3])]

        metadata = {
            "camera": frame_i.camera,
            "camera_quality_score": torch.tensor([frame_i.camera_quality_score]).reshape((-1, 1)),
            "crop_bbox": crop_bbox,
            "frame_timestamp": torch.tensor([frame_i.frame_timestamp]).reshape((-1, 1)),
        }
        return img_, mask_, metadata

    def get_masks(self, list_items: list[int]) -> np.ndarray:
        """
        We refined the original masks from CoP3D and provide a download link to add them to external_data folder.
        We strongly recommend using the refined masks instead of the original ones from Cop3D.

        Return:
            - mask_list [N,H,W,1] NPFLOAT32 [0,1] (fg probability)
        """
        # Try to get the refined masks
        path_refined_masks = os.path.join(Keys().source_refined_masks, self.sequence_index)

        if not os.path.isdir(path_refined_masks):
            print(f"WARNING: refined masks for CoP3D sequence {self.sequence_index} not found in {path_refined_masks}.")
            print("This will affect result quality.")
            # Get classical masks
            mask_list = [np.transpose(self.__getitem__(i)[1].numpy().astype(np.float32), (0, 2, 3, 1)) for i in list_items]
        else:
            # Get refined masks
            list_frames = [os.path.join(path_refined_masks, "frame{:06d}.png".format(self.sequence_frame_ids[i])) for i in list_items]
            mask_list = []
            for path_i in list_frames:
                X = np.expand_dims(np.array(Image.open(path_i)).astype(np.uint8), 0)  # np.expand_dims adds C dimension, X is now (C,H,W)=(1,488,363) -> required for resize_image // values of X are 0 or 1 for refined masks.
                if self.cop3d_resizing:
                    X_resized = resize_image(X, 800, 800, mode="bilinear")[0].unsqueeze(0).numpy() # X_resized is (1,1,800,800) (one more 1 is added by .unsqueeze(0))
                    mask_list.append(np.transpose(X_resized.astype(np.float32), (0, 2, 3, 1))) # reordering dimensions to (1,800,800,1)
                else:
                    X = np.expand_dims(X, 0)
                    mask_list.append(np.transpose(X.astype(np.float32), (0, 2, 3, 1)))

        mask_list = array_list_to_stack(mask_list)
        return mask_list # mask_list is of shape (202, 483, 360, 1)

    def get_imgs_rgb(self, list_items: list[int]) -> np.ndarray:
        # list_items is list of ints [0,1,2,3 ---]
        """
        Return:
            - img_list [N,H,W,3] NPFLOAT32 [0,1]
        """
        img_list = [np.transpose(self.__getitem__(i)[0].numpy().astype(np.float32), (0, 2, 3, 1)) for i in list_items]
        img_list = array_list_to_stack(img_list)
        return img_list # image is (N,H,W,3), img values are in [0,1]

    def get_cameras(self, list_items: list[int]) -> PerspectiveCameras:
        cameras = [self.__getitem__(i)[2]["camera"] for i in list_items]
        # torch.save(cameras, "cop3d_data/dog/565_81664_160332/cameras/cameras.pt") # NOTE make sure all list_items includes to entire sequence!
        return join_cameras_as_batch(cameras)

    def get_sparse_keypoints(self, list_items: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            - sparse_keypoints (N, N_KPTS, 2) NPFLOAT32
            - scores (N, N_KPTS, 1) NPFLOAT32
        """
        sparse_keypoints_path = os.path.join(Keys().source_refined_keypoints, f"{self.sequence_index}_keypoints.pk")
        assert os.path.isfile(sparse_keypoints_path), (
            f"No keypoints data found for sequence {self.sequence_index} in: {sparse_keypoints_path},",
            'if no sparse keypoints available, please set "exp.l_optim_sparse_kp=0" in config/config.yaml',
            "This will affect result quality.",
        )

        with open(sparse_keypoints_path, "rb") as f:
            X = pk.load(f)
            sparse_keypoints, scores = X[..., :2], X[..., 2:]

        return sparse_keypoints, scores

    def get_init_shape(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            - init_betas NPFLOAT32
            - init_betas_limbs NPFLOAT32
        """
        init_shape_path = os.path.join(Keys().source_init_shape, f"{self.sequence_index}_init_pose.pk")

        with open(init_shape_path, "rb") as f:
            X = pk.load(f)
            init_betas, init_betas_limbs = X["betas"], X["betas_limbs"]

        return init_betas, init_betas_limbs

    def get_crop_bbox_list(self, list_items=None):
        if list_items is None:
            crop_bbox_list = [self.__getitem__(i)[2]["crop_bbox"] for i in range(len(self))]
        else:
            crop_bbox_list = [self.__getitem__(i)[2]["crop_bbox"] for i in list_items]
        return crop_bbox_list
