import cv2
import torch
import numpy as np


def plot_liteboxnet_label(image, label, threshold=0.5):
    cv_im = np.transpose(image.numpy(), (1, 2, 0))
    min_value = cv_im.min()
    max_value = cv_im.max()
    cv_im = ((cv_im - min_value) / (max_value - min_value) * 255).astype(np.uint8)
    cv_im = np.ascontiguousarray(cv_im)

    im_h = cv_im.shape[0]
    im_w = cv_im.shape[1]

    cell_h = cv_im.shape[0] / label.shape[1]
    cell_w = cv_im.shape[1] / label.shape[2]

    for i in range(label.shape[1]):
        for j in range(label.shape[2]):
            car_slice = label[:, i, j]
            if car_slice[0] > threshold:
                x_center = int((car_slice[2] + j) * cell_w)
                y_center = int((car_slice[1] + i) * cell_h)
                x1, y1 = x_center + car_slice[5] * (car_slice[3] * im_w / 2), y_center + car_slice[4] * (
                        car_slice[3] * im_w / 2)
                x2, y2 = x_center - car_slice[5] * (car_slice[3] * im_w / 2), y_center - car_slice[4] * (
                        car_slice[3] * im_w / 2)
                x3, y3 = x_center + car_slice[8] * (car_slice[6] * im_w / 2), y_center + car_slice[7] * (
                        car_slice[6] * im_w / 2)
                x4, y4 = x_center - car_slice[8] * (car_slice[6] * im_w / 2), y_center - car_slice[7] * (
                        car_slice[6] * im_w / 2)

                cv2.line(cv_im, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
                cv2.line(cv_im, (int(x3), int(y3)), (int(x4), int(y4)), (255, 0, 0), 1)
                cv2.line(cv_im, (int(x_center), int(y_center)), (int(x_center), int(y_center - car_slice[9] * im_h)),
                         (0, 255, 0), 1)

    return cv_im.astype(int)


def read_label_file(label_file_path):
    line_list = []
    with open(label_file_path) as f:
        for line in f:
            line = line.split()
            line_list.append(line)

    # each line corresponds to one detection
    det_dict_list = []
    for line in line_list:
        # det_dict holds info on one detection
        det_dict = {
            'class': float(line[0]),
            'y': float(line[1]),
            'x': float(line[2]),
            'l1': float(line[3]),
            'sin1': float(line[4]),
            'cos1': float(line[5]),
            'l2': float(line[6]),
            'sin2': float(line[7]),
            'cos2': float(line[8]),
            'height': float(line[9])
        }
        det_dict_list.append(det_dict)
    return det_dict_list


def dict_list_to_label(det_dict_list, image_size, label_size, vehicle_confidence_multiplier):
    label = np.zeros((10, *label_size), dtype=float)
    for det_dict in det_dict_list:
        if not (0 <= det_dict["y"] <= image_size[0] and 0 <= det_dict["x"] <= image_size[1] and 0 <= det_dict["l1"] <=
                image_size[1] and 0 <= det_dict["l2"] <= image_size[1] and 0 <= det_dict["height"] <= image_size[0]):
            continue
        y = (det_dict["y"] * label_size[0]) / image_size[0]
        y_label = int(y)
        y_relative_to_grid_cell = y - int(y)

        x = (det_dict["x"] * label_size[1]) / image_size[1]
        x_label = int(x)
        x_relative_to_grid_cell = x - int(x)

        l1_relative_to_image = det_dict['l1'] / image_size[1]
        l2_relative_to_image = det_dict['l2'] / image_size[1]
        height_relative_to_image = det_dict["height"] / image_size[0]

        kernel_size = vehicle_confidence_multiplier.shape

        label_y_range = (max(0, y_label - kernel_size[0] // 2), min(label_size[0], y_label + kernel_size[0] // 2 + 1))
        label_x_range = (max(0, x_label - kernel_size[1] // 2), min(label_size[1], x_label + kernel_size[1] // 2 + 1))
        kernel_y_range = (max(0, kernel_size[0] // 2 - y_label), min(kernel_size[0], label_size[0] - y_label + kernel_size[0] // 2))
        kernel_x_range = (max(0, kernel_size[1] // 2 - x_label), min(kernel_size[1], label_size[1] - x_label + kernel_size[1] // 2))

        if det_dict["class"] >= 0 and det_dict["height"] >= 30:
            label[
                0,
                label_y_range[0]:label_y_range[1],
                label_x_range[0]:label_x_range[1]
            ] = np.maximum(
                label[
                    0,
                    label_y_range[0]:label_y_range[1],
                    label_x_range[0]:label_x_range[1]
                ],
                det_dict["class"] * vehicle_confidence_multiplier[
                    kernel_y_range[0]:kernel_y_range[1],
                    kernel_x_range[0]:kernel_x_range[1]
                ]
            )
        else:
            label[
                0,
                label_y_range[0]:label_y_range[1],
                label_x_range[0]:label_x_range[1]
            ] = -1 * np.ones(kernel_size)[
                kernel_y_range[0]:kernel_y_range[1],
                kernel_x_range[0]:kernel_x_range[1]
            ]

        label[1:, y_label, x_label] = np.array([y_relative_to_grid_cell, x_relative_to_grid_cell,
                                                l1_relative_to_image, det_dict['sin1'], det_dict['cos1'],
                                                l2_relative_to_image, det_dict['sin2'], det_dict['cos2'],
                                                height_relative_to_image])
    label[0, :, :] = np.clip(label[0, :, :], a_min=-1, a_max=1)
    return label.astype(float)
