# create backup
cp cop3d_data/metadata.sqlite cop3d_data/metadata.sqlite.bak

# update database
sqlite3 cop3d_data/metadata.sqlite "UPDATE frame_annots SET _image_size = X'e301000068010000' WHERE _image_path LIKE '%dog/565_81664_160332/%';"