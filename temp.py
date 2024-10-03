import os
import shutil

out_path = "/run/user/1000/gvfs/smb-share:server=wdmycloudex4100.local,share=public/CSAW Dataset/2021-204-1-1/"
source_path = "/run/user/1000/gvfs/smb-share:server=wdmycloudex4100.local,share=public/CSAW Dataset/2021-204-1-1/data/"
categories = ["L_MLO", "R_MLO", "L_CC", "R_CC"]

for cat in categories:
    if not os.path.exists(os.path.join(out_path, f"{cat}/")):
        os.mkdir(os.path.join(out_path, f"{cat}/"))

images_list = os.listdir(source_path)

for image_id in images_list:

    print(image_id)
    path_to_current_image = os.path.join(source_path, image_id)

    for cat in categories:

        destination_path = os.path.join(out_path, f"{cat}/", image_id)

        if cat in image_id:
            shutil.move(path_to_current_image, destination_path)
            continue
