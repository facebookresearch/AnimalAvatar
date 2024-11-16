# Animal Avatar: Reconstructing Animatable 3D Animals from Casual Videos

Code from the **ECCV 2024 (Selected as an Oral)** paper: 

**Animal Avatar: Reconstructing Animatable 3D Animals from Casual Videos.**

[Remy Sabathier](https://profiles.ucl.ac.uk/96179-remy-sabathier)<sup>1,2</sup>, 
[Niloy J. Mitra](http://www0.cs.ucl.ac.uk/staff/n.mitra/)<sup>2</sup>, 
[David Novotny](https://d-novotny.github.io/)<sup>1</sup>,

<sup>1</sup>Meta London, <sup>2</sup>University College London (UCL)

<img src="docs/teaser.png" style="width: 60%;">

| [**Project Page**](https://remysabathier.github.io/animalavatar.github.io/) | [**ArXiv**](https://arxiv.org/abs/2403.17103) | [**Code**](https://github.com/facebookresearch/AnimalAvatar?tab=readme-ov-file) | [**ECCV Oral**](https://eccv2024.ecva.net/virtual/2024/oral/1988) |


## News
[**12.08.2024**] &#x1F389; **AnimalAvatar** is selected as an oral at ECCV 2024 ! Info [here](https://eccv2024.ecva.net/virtual/2024/oral/1988) <br>
[**30.07.2024**] Initial code release <br>
[**01.07.2024**] &#x1F389; **AnimalAvatar** is accepted at [ECCV 2024](https://eccv.ecva.net/) ! <br>


## Installation

The code was developed with python=3.10 | pytorch=2.0.1 | pytorch-cuda=11.8 <br>
Install the following libs:
1) [Install PyTorch](https://pytorch.org/get-started/locally/#start-locally).
2) [Install PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
3) [Install Lightplane](https://github.com/facebookresearch/lightplane?tab=readme-ov-file#installation)
4) [Install DensePoseCSE](https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/doc/GETTING_STARTED.md#installation-as-a-package)

Additional pip install:

```python
pip install pandas sqlalchemy plotly hydra-core tensorboard lpips opencv-python imageio[ffmpeg]
```

Please download [external_data/](https://github.com/RemySabathier/animalavatar.github.io/raw/main/external_data/external_data.zip) and add the following files to "external_data/" folder:

Download SMAL model from ["BITE: Beyond priors for improved three-D dog pose estimation"](https://github.com/runa91/bite_release):
* [my_smpl_39dogsnorm_newv3_dog.pkl](https://owncloud.tuebingen.mpg.de/index.php/s/BpPWyzsmfycXdyj/download?path=%2Fdata%2Fsmal_data%2Fnew_dog_models&files=my_smpl_39dogsnorm_newv3_dog.pkl&downloadStartSecret=21p5mlf8old)
* [symmetry_inds.json](https://owncloud.tuebingen.mpg.de/index.php/s/BpPWyzsmfycXdyj/download?path=%2Fdata%2Fsmal_data&files=symmetry_inds.json&downloadStartSecret=ecjw1bt2rbv)

Download Densepose model weight and configs:
* [densepose_rcnn_R_50_FPN_soft_animals_I0_finetune_i2m_16k.yaml](https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/DensePose/configs/cse/densepose_rcnn_R_50_FPN_soft_animals_I0_finetune_i2m_16k.yaml)
* [Base-DensePose-RCNN-FPN.yaml](https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/DensePose/configs/cse/Base-DensePose-RCNN-FPN.yaml)
* [model_final_8c9d99.pkl](https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_animals_I0_finetune_i2m_16k/270727461/model_final_8c9d99.pkl)


External data folder should look as follows:

```bash
├── external_data/
│   ├── cse/
│   │   ├── Base-DensePose-RCNN-FPN.yaml
│   │   ├── cse_embedding.pk
│   │   ├── densepose_rcnn_R_50_FPN_soft_animals_I0_finetune_i2m_16k.yaml
│   │   └── model_final_8c9d99.pkl
│   ├── init_pose/
│   │   ├── ...
│   ├── lbos/
│   │   ├── lbo_cse_to_smal.pk
│   │   ├── lbo_cse.pk
│   │   └── lbo_smal.pk
│   ├── refined_masks/
│   │   ├── ...
│   ├── smal/
│   │   ├── my_smpl_39dogsnorm_newv3_dog.pkl
│   │   └── symmetry_inds.json
│   ├── sparse_keypoints/
│   │   ├── ...
│   ├── textures/
│   │   ├── texture_cse.pk
│   │   └── texture_smal.pk
```

The project was developed for the [CoP3D](https://github.com/facebookresearch/cop3d) dataset. Follow [instructions](https://github.com/facebookresearch/cop3d#download) to download CoP3D.

## Set-up paths

In **config/keys.py**, manually enter the path where you store external data, and the path of the CoP3D dataset on your machine.

## Optimize a CoP3D scene

<img src="docs/preprocess.png" style="width: 60%;">

### 1- Preprocessing

Before reconstructing a scene, you must preprocess the video sequence to extract a **CSE map** and a **root orientation** per frame.

```python
#Example code to process the CoP3D sequence "1030_23106_17099"
python main_preprocess_scene.py -sequence_index "1030_23106_17099" --visualize
```

A visualization of the processed CSE map and root orientation is saved in the preprocessing folder. For a subset of CoP3D scenes (list available in "config/keys.py"), we provide in **external_data/** refined masks, init shape and sparse keypoints per frame.

### 2- Launch Reconstruction (Optimizer)

The optimization framework **SceneOptimizer** propose a general framework to optimize the reconstruction of a scene given multiple modalities. Our implementation, aligned with **AnimalAvatar**, contains several losses available in **scene_optim/losses_optim**.


To launch the optimization on a CoP3D scene:

```python
#Example code to optimize the CoP3D sequence "1030_23106_17099" and save it in folder experiments/
python main_optimize_scene.py 'exp.sequence_index="1030_23106_17099"' 'exp.experiment_folder="experiments"'
```

Parameters of the reconstruction are accessible in the config file config/config.yaml

### 3- Visualize Reconstruction

To visualize reconstruction of a trained model:

```python
python main_visualize_reconstruction.py  "path_of_the_reconstruction_folder"
```

## Optimize a Custom scene

**AnimalAvatar** scene-optimizer relies on the following inputs:
  - Ground-truth RGB images
  - Ground-truth masks
  - Ground-truth cameras (intrinsics & extrinsics)

To launch **AnimalAvatar** on a custom video, fill **CustomSingleVideo** in *data/custom_dataloader.py* with your data, and launch AnimalAvatar with 'CUSTOM' dataset option:

```python
# 1- Preprocess the custom scene (to get CSE map and root orientation per frame)
python main_preprocess_scene.py -sequence_index "XXX_XXX_XXX" --custom --visualize
# 2- Launch the reconstruction
python main_optimize_scene.py 'exp.sequence_index="XXX_XXX_XXX"' 'exp.dataset_source="CUSTOM"' 'exp.l_optim_sparse_kp=0'
# 3- Visualize the reconstruction
python main_visualize_reconstruction.py "path_of_the_custom_reconstruction_folder" --custom
```

* If your scene is missing cameras, we recommend using [VGGSfM](https://github.com/facebookresearch/vggsfm). <br>
* If your scene is missing masks, we recommend using [Segment-Anything](https://github.com/facebookresearch/segment-anything) <br>

Code will be added to help through this process.

## TODO

* Include Sparse-keypoint keypoint predictor from [BITE](https://github.com/runa91/bite_release)
* Add CSE embedding refinement with [Optical-Flow](https://github.com/princeton-vl/RAFT)


## License

See the LICENSE file for details about the license under which this code is made available.


## Citation

```
@inproceedings{AnimalAvatars2024,
author = {Sabathier, Remy and Mitra, Niloy J. and Novotny, David},
title = {Animal Avatars: Reconstructing Animatable 3D Animals from Casual Videos},
year = {2024},
booktitle = {Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part LXXIX},
pages = {270–287},
doi = {10.1007/978-3-031-72986-7_16}
}
```