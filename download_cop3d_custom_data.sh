####### images
# curl -L https://nextcloud.in.tum.de/index.php/s/XAxFcXryrBYMJQM/download/dog_cop3d_565_81664_160332_images_custom.zip -o cop3d_data_images_custom.zip
# unzip cop3d_data_images_custom.zip
# rm cop3d_data_images_custom.zip

## backup existing images
# mv cop3d_data/dog/565_81664_160332/images cop3d_data/dog/565_81664_160332/images_backup

## move custom images to the correct location
# mv dog_cop3d_565_81664_160332_custom cop3d_data/dog/565_81664_160332/images


####### masks
# curl -L https://nextcloud.in.tum.de/index.php/s/ESEKgByHb6mn7Ns/download/dog_cop3d_565_81664_160332_masks_custom_animal_avatar.zip -o cop3d_data_masks_custom.zip
# unzip cop3d_data_masks_custom.zip
# rm cop3d_data_masks_custom.zip

## backup existing masks
# mv cop3d_data/dog/565_81664_160332/masks cop3d_data/dog/565_81664_160332/masks_backup

## move custom masks to the correct location
# mv dog_cop3d_565_81664_160332_custom_animal_avatar cop3d_data/dog/565_81664_160332/masks

####### cameras
# curl -L https://nextcloud.in.tum.de/index.php/s/7nC2H7xrg64cjAe/download/cameras.pt -o cameras.pt
# mkdir cop3d_data/dog/565_81664_160332/cameras
# mv cameras.pt cop3d_data/dog/565_81664_160332/cameras

####### refined masks cropped
# curl -L https://nextcloud.in.tum.de/index.php/s/8AoFJsRGaMkWqeX/download/565_81664_160332_refined_masks_cropped.zip -o 565_81664_160332_refined_masks_cropped.zip
# unzip 565_81664_160332_refined_masks_cropped.zip
# rm 565_81664_160332_refined_masks_cropped.zip

## backup existing masks
# mv external_data/refined_masks/565_81664_160332 external_data/refined_masks/565_81664_160332_backup

## move custom masks to the correct location
# mv 565_81664_160332 external_data/refined_masks/565_81664_160332 external_data/refined_masks

