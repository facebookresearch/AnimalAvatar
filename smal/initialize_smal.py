import torch
import json
import numpy as np
import pickle as pk
from scipy.sparse import csc_matrix
from utils.torch_util import SuppressAllWarnings

"""
Initialize SMAL model from the SMAL archive from
 "BITE: Beyond priors for improved three-D dog pose estimation", 
    N Rüegg, S Tripathi, K Schindler, MJ Black, S Zuffi
https://github.com/runa91/bite_release
"""

CANONICAL_MODEL_JOINTS_REFINED = [
    41,
    9,
    8,
    43,
    19,
    18,
    42,
    13,
    12,
    44,
    23,
    22,
    25,
    31,
    33,
    34,
    35,
    36,
    38,
    37,
    39,
    40,
    46,
    45,
    28,
]


def get_beta_scale_mask() -> torch.Tensor:
    """
    See BITE code: https://github.com/runa91/bite_release .
    """

    part_list = ["legs_l", "legs_f", "tail_l", "tail_f", "ears_y", "ears_l", "head_l"]
    n_b_log = len(part_list)

    beta_scale_mask = torch.zeros(35, 3, n_b_log)

    # which joints belong to which bodypart
    leg_joints = list(range(7, 11)) + list(range(11, 15)) + list(range(17, 21)) + list(range(21, 25))
    tail_joints = list(range(25, 32))
    ear_joints = [33, 34]
    mouth_joints = [16, 32]

    for ind, part in enumerate(part_list):
        if part == "legs_l":
            beta_scale_mask[leg_joints, [2], [ind]] = 1.0  # Leg lengthening
        elif part == "legs_f":
            beta_scale_mask[leg_joints, [0], [ind]] = 1.0  # Leg fatness
            beta_scale_mask[leg_joints, [1], [ind]] = 1.0  # Leg fatness
        elif part == "tail_l":
            beta_scale_mask[tail_joints, [0], [ind]] = 1.0  # Tail lengthening
        elif part == "tail_f":
            beta_scale_mask[tail_joints, [1], [ind]] = 1.0  # Tail fatness
            beta_scale_mask[tail_joints, [2], [ind]] = 1.0  # Tail fatness
        elif part == "ears_y":
            beta_scale_mask[ear_joints, [1], [ind]] = 1.0  # Ear y
        elif part == "ears_l":
            beta_scale_mask[ear_joints, [2], [ind]] = 1.0  # Ear z
        elif part == "head_l":
            beta_scale_mask[mouth_joints, [0], [ind]] = 1.0  # Head lengthening
        else:
            print(part + " not available")
            raise ValueError

    beta_scale_mask = torch.transpose(beta_scale_mask.reshape(35 * 3, n_b_log), 0, 1)
    return beta_scale_mask


def get_symmetry_inds(symmetry_json_path: str) -> tuple[torch.Tensor, int, int, int]:
    """
    See BITE code: https://github.com/runa91/bite_release .

    Returns:
        - inds_back (3889,) TORCHINT64
        - n_center, n_left, sl
    """

    with open(symmetry_json_path, "r") as f:
        sym_ids_dict = json.load(f)

    sym_center_ids = np.asarray(sym_ids_dict["center_inds"])
    sym_left_ids = np.asarray(sym_ids_dict["left_inds"])
    sym_right_ids = np.asarray(sym_ids_dict["right_inds"])

    n_center = sym_center_ids.shape[0]
    n_left = sym_left_ids.shape[0]
    n_right = sym_right_ids.shape[0]
    sl = 2 * n_center

    inds_back = np.zeros((3889,), dtype=np.int64)

    for ind in range(n_center):
        ind_in_forward = sym_center_ids[ind]
        inds_back[ind_in_forward] = ind

    for ind in range(n_left):
        ind_in_forward = sym_left_ids[ind]
        inds_back[ind_in_forward] = sym_center_ids.shape[0] + ind

    for ind in range(n_right):
        ind_in_forward = sym_right_ids[ind]
        inds_back[ind_in_forward] = sym_center_ids.shape[0] + sym_left_ids.shape[0] + ind

    return torch.Tensor(inds_back).type(torch.int64), n_center, n_left, sl


def process_original_smal(path_original: str, symmetry_json_path: str) -> dict[str, torch.Tensor | int]:
    """
    Preprocess SMAL model from:
        "BITE: Beyond priors for improved three-D dog pose estimation",
            N Rüegg, S Tripathi, K Schindler, MJ Black, S Zuffi
        source code: "https://github.com/runa91/bite_release"
    """
    save_dict = {}

    with SuppressAllWarnings():
        with open(path_original, "rb") as f:
            X_ori = pk.load(f)

    inds_back, n_center, n_left, sl = get_symmetry_inds(symmetry_json_path)

    save_dict["betas_scale_mask"] = get_beta_scale_mask()

    "1) faces (7774, 3)"
    save_dict["faces"] = torch.Tensor(X_ori["f"]).type(torch.int64)

    "2) v_template (3889, 3)"
    save_dict["v_template"] = torch.Tensor(X_ori["v_template"]).type(torch.float32)

    "3) shapedirs (78, 11667)"
    num_betas = X_ori["shapedirs"].shape[-1]
    shapedirs = X_ori["shapedirs"].reshape(-1, num_betas).T
    save_dict["shapedirs"] = torch.Tensor(shapedirs).type(torch.float32)

    "4) J_regressor (3889, 35)"
    joint_regressor = X_ori["J_regressor"].T.todense()
    save_dict["joint_regressor"] = torch.Tensor(joint_regressor).type(torch.float32)

    "5) posedirs (306, 11667)"
    num_pose_basis = X_ori["posedirs"].shape[-1]
    posedirs = X_ori["posedirs"].reshape(-1, num_pose_basis).T
    save_dict["posedirs"] = torch.Tensor(posedirs).type(torch.float32)

    "6) kintree_table (35)"
    kintree_table = X_ori["kintree_table"][0].astype(np.int32)
    save_dict["kintree_table"] = torch.Tensor(kintree_table).type(torch.int32)

    "7) weights (3889, 35)"
    save_dict["weights"] = torch.Tensor(X_ori["weights"]).type(torch.float32)

    "8) inds_back (3889)"
    save_dict["inds_back"] = inds_back

    "9) n_center (int)"
    save_dict["n_center"] = n_center
    "10) n_left (int)"
    save_dict["n_left"] = n_left
    "11) s_left (int)"
    save_dict["s_left"] = sl

    return save_dict
