#!/bin/bash

# Download and extract external_data.zip
curl -L https://github.com/RemySabathier/animalavatar.github.io/raw/main/external_data/external_data.zip -o external_data.zip
unzip external_data.zip
rm external_data.zip

# Change to external_data directory
cd ./external_data

# Download SMAL dog model
curl -L "https://owncloud.tuebingen.mpg.de/index.php/s/BpPWyzsmfycXdyj/download?path=%2Fdata%2Fsmal_data%2Fnew_dog_models&files=my_smpl_39dogsnorm_newv3_dog.pkl&downloadStartSecret=21p5mlf8old" -o ./smal/my_smpl_39dogsnorm_newv3_dog.pkl

# Download symmetry indices
curl -L "https://owncloud.tuebingen.mpg.de/index.php/s/BpPWyzsmfycXdyj/download?path=%2Fdata%2Fsmal_data&files=symmetry_inds.json&downloadStartSecret=ecjw1bt2rbv" -o ./smal/symmetry_inds.json

# Download DensePose config files and model
curl -L "https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/DensePose/configs/cse/densepose_rcnn_R_50_FPN_soft_animals_I0_finetune_i2m_16k.yaml" -o ./cse/densepose_rcnn_R_50_FPN_soft_animals_I0_finetune_i2m_16k.yaml
curl -L "https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/DensePose/configs/cse/Base-DensePose-RCNN-FPN.yaml" -o ./cse/Base-DensePose-RCNN-FPN.yaml
curl -L "https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_animals_I0_finetune_i2m_16k/270727461/model_final_8c9d99.pkl" -o ./cse/model_final_8c9d99.pkl

echo "All downloads completed successfully."