# images
curl -L https://nextcloud.in.tum.de/index.php/s/XAxFcXryrBYMJQM/download/dog_cop3d_565_81664_160332_images_custom.zip -o cop3d_data_images_custom.zip
unzip cop3d_data_images_custom.zip
rm cop3d_data_images_custom.zip

# backup existing images
mv cop3d_data/dog/565_81664_160332/images cop3d_data/dog/565_81664_160332/images_backup

# move custom images to the correct location
mv dog_cop3d_565_81664_160332_custom cop3d_data/dog/565_81664_160332/images


# masks
curl -L https://nextcloud.in.tum.de/index.php/s/ESEKgByHb6mn7Ns/download/dog_cop3d_565_81664_160332_masks_custom_animal_avatar.zip -o cop3d_data_masks_custom.zip
unzip cop3d_data_masks_custom.zip
rm cop3d_data_masks_custom.zip

# backup existing masks
mv cop3d_data/dog/565_81664_160332/masks cop3d_data/dog/565_81664_160332/masks_backup

# move custom masks to the correct location
mv dog_cop3d_565_81664_160332_custom_animal_avatar cop3d_data/dog/565_81664_160332/masks
