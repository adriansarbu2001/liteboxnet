import os
import cv2
import math
import numpy as np
from tqdm import tqdm

from liteboxnet.dataset.kitti_dataset import KittiDataset
from liteboxnet.utils.kitti_utils import get_coords_3d


def resize_images(input_folder, output_folder, target_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    progress_bar = tqdm(total=len(os.listdir(input_folder)), desc="Processing Images", unit="image")
    
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            img = cv2.imread(input_path)
            resized_img = cv2.resize(img, target_size)
            cv2.imwrite(output_path, resized_img)
            
        progress_bar.update(1)

    progress_bar.close()


def modify_labels(input_root_folder, output_root_folder, split, target_size):
    if not os.path.exists(output_root_folder):
        os.makedirs(output_root_folder)

    input_dataset = KittiDataset(
        base_root=input_root_folder,
        split=split
    )
    output_root_folder = os.path.join(output_root_folder, split)

    # in_images_folder = os.path.join(input_root_folder, "image_2")
    # in_labels_folder = os.path.join(input_root_folder, "label_2")
    # in_calib_folder = os.path.join(input_root_folder, "calib")
    out_images_folder = os.path.join(output_root_folder, "image_2")
    out_labels_folder = os.path.join(output_root_folder, "label_2")

    if not os.path.exists(out_images_folder):
        os.makedirs(out_images_folder)
    if not os.path.exists(out_labels_folder):
        os.makedirs(out_labels_folder)

    progress_bar = tqdm(total=len(input_dataset), desc="Processing Labels", unit="label")
    
    for index in range(len(input_dataset)):
    # for index in [27]:
        calib, image, label, meta = input_dataset[index]

        with open(os.path.join(out_labels_folder, meta['label_name']), 'w') as output_file:
            for i in range(0, len(label)):
                if label[i]['pos'][2] > 2 and label[i]['truncation'] < 1:
                    cls = label[i]['class']
                    if (cls == "Car" or cls == "Van") and label[i]['occlusion'] <= 1:
                        cls = 1
                    else:
                        cls = -1
                    coords, _, _ = get_coords_3d(label[i], calib)
                    coords = np.transpose(coords)
                    coords[:, 0] *= (target_size[0] / image.shape[1])
                    coords[:, 1] *= (target_size[1] / image.shape[0])
                    x1, y1 = ((coords[0, 0] + coords[1, 0]) / 2), ((coords[0, 1] + coords[1, 1]) / 2)
                    x2, y2 = ((coords[2, 0] + coords[3, 0]) / 2), ((coords[2, 1] + coords[3, 1]) / 2)
                    x3, y3 = ((coords[0, 0] + coords[3, 0]) / 2), ((coords[0, 1] + coords[3, 1]) / 2)
                    x4, y4 = ((coords[1, 0] + coords[2, 0]) / 2), ((coords[1, 1] + coords[2, 1]) / 2)
                    x5, y5 = ((coords[4, 0] + coords[5, 0]) / 2), ((coords[4, 1] + coords[5, 1]) / 2)
                    x6, y6 = ((coords[6, 0] + coords[7, 0]) / 2), ((coords[6, 1] + coords[7, 1]) / 2)
                    x7, y7 = ((coords[4, 0] + coords[7, 0]) / 2), ((coords[4, 1] + coords[7, 1]) / 2)
                    x8, y8 = ((coords[5, 0] + coords[6, 0]) / 2), ((coords[5, 1] + coords[6, 1]) / 2)

                    x_intersection = x4 + (x3 - x4) / 2 if x4 < x3 else x3 + (x4 - x3) / 2
                    y_intersection = y4 + (y3 - y4) / 2 if y4 < y3 else y3 + (y4 - y3) / 2
                    l1 = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    l2 = math.sqrt((x4 - x3)**2 + (y4 - y3)**2)
                    sin1 = (y1 - y_intersection) / (l1 / 2)
                    sin2 = (y3 - y_intersection) / (l2 / 2)
                    cos1 = (x1 - x_intersection) / (l1 / 2)
                    cos2 = (x3 - x_intersection) / (l2 / 2)
                    height = (math.sqrt((x5 - x1)**2 + (y5 - y1)**2) + math.sqrt((x6 - x2)**2 + (y6 - y2)**2)) / 2
                    # print(cls, y_intersection, x_intersection, l1, sin1, cos1, l2, sin2, cos2, height)
                    output_file.write(f"{cls} {y_intersection} {x_intersection} {l1} {sin1} {cos1} {l2} {sin2} {cos2} {height}\n")
            
        progress_bar.update(1)

    progress_bar.close()


if __name__ == '__main__':
    input_image_folder = 'D:/UBB/Inteligenta Computationala Aplicata EN/SEM I/datasets/kitti/training/image_2'
    output_image_folder = 'D:/UBB/Inteligenta Computationala Aplicata EN/SEM I/datasets/liteboxnet/training/image_2'
    
    in_root_folder = 'D:/UBB/Inteligenta Computationala Aplicata EN/SEM I/datasets/kitti'
    out_root_folder = 'D:/UBB/Inteligenta Computationala Aplicata EN/SEM I/datasets/liteboxnet'

    # resize_images(input_image_folder, output_image_folder, target_size=(1224, 370))
    # modify_labels(in_root_folder, out_root_folder, "training", target_size=(1224, 370))
    resize_images(input_image_folder, output_image_folder, target_size=(1216, 352))
    modify_labels(in_root_folder, out_root_folder, "training", target_size=(1216, 352))
