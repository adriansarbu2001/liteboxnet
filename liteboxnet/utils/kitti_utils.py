import os
import time
import matplotlib.pyplot as plt
import numpy as np

import cv2
import PIL
from PIL import Image
from math import cos,sin


def pil_to_cv(pil_im):
    """ convert PIL image to cv2 image"""
    open_cv_image = np.array(pil_im) 
    # Convert RGB to BGR 
    return open_cv_image[:, :, ::-1] 


def get_coords_3d(det_dict, P):
    """ returns the pixel-space coordinates of an object's 3d bounding box
        computed from the label and the camera parameters matrix
        for the idx object in the current frame
        det_dict - object representing one detection
        P - camera calibration matrix
        bbox3d - 8x2 numpy array with x,y coords for ________ """     
    # create matrix of bbox coords in physical space 

    l = det_dict['dim'][0]
    w = det_dict['dim'][1]
    h = det_dict['dim'][2]
    x_pos = det_dict['pos'][0]
    y_pos = det_dict['pos'][1]
    z_pos = det_dict['pos'][2]
    ry = det_dict['rot_y']
    cls = det_dict['class']
        
        
    # in absolute space (meters relative to obj center)
    obj_coord_array = np.array([[  l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2],
                                [    0,    0,    0,    0,   -h,   -h,   -h,   -h],
                                [  w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]])
    
    # apply object-centered rotation here
    R = np.array([[  cos(ry),        0,  sin(ry)],
                  [        0,        1,        0],
                  [ -sin(ry),        0,  cos(ry)]])
    rotated_corners = np.matmul(R, obj_coord_array)
    
    rotated_corners[0,:] += x_pos
    rotated_corners[1,:] += y_pos
    rotated_corners[2,:] += z_pos
    
    # transform with calibration matrix
    # add 4th row for matrix multiplication
    zeros = np.zeros([1,np.size(rotated_corners,1)])
    rotated_corners = np.concatenate((rotated_corners,zeros),0)

    
    pts_2d = np.matmul(P,rotated_corners)
    pts_2d[0,:] = pts_2d[0,:] / pts_2d[2,:]        
    pts_2d[1,:] = pts_2d[1,:] / pts_2d[2,:] 
    
    # apply camera space rotation here?
    return pts_2d[:2,:] ,pts_2d[2,:], rotated_corners

    
def draw_prism(image, coords, color):
    """ draws a rectangular prism on a copy of an image given the x,y coordinates 
    of the 8 corner points, does not make a copy of original image
    image - cv2 image
    coords - 2x8 numpy array with x,y coords for each corner
    prism_im - cv2 image with prism drawn"""
    prism_im = image.copy()
    coords = np.transpose(coords).astype(int)
    #fbr,fbl,rbl,rbr,ftr,ftl,frl,frr
    edge_array= np.array([[ 0, 1, 0, 1, 1, 0, 0, 0 ],
                          [ 1, 0, 1, 0, 0, 1, 0, 0 ],
                          [ 0, 1, 0, 1, 0, 0, 1, 1 ],
                          [ 1, 0, 1, 0, 0, 0, 1, 1 ],
                          [ 1, 0, 0, 0, 0, 1, 0, 1 ],
                          [ 0, 1, 0, 0, 1, 0, 1, 0 ],
                          [ 0, 0, 1, 0, 0, 1, 0, 1 ],
                          [ 0, 0, 0, 1, 1, 0, 1, 0 ]])

    # plot lines between indicated corner points
    for i in range(0, 8):
        for j in range(0, 8):
            if edge_array[i, j] == 1:
                cv2.line(prism_im, (coords[i, 0], coords[i, 1]), (coords[j, 0], coords[j, 1]), color, 1)
    return prism_im

    
def draw_base(image, coords, color):
    """ draws a rectangular prism on a copy of an image given the x,y coordinates 
    of the 8 corner points, does not make a copy of original image
    image - cv2 image
    coords - 2x8 numpy array with x,y coords for each corner
    prism_im - cv2 image with prism drawn"""
    prism_im = image.copy()
    coords = np.transpose(coords).astype(int)
    #fbr,fbl,rbl,rbr,ftr,ftl,frl,frr
    edge_array= np.array([[ 0, 1, 0, 1, 0, 0, 0, 0 ],
                          [ 1, 0, 1, 0, 0, 0, 0, 0 ],
                          [ 0, 1, 0, 1, 0, 0, 0, 0 ],
                          [ 1, 0, 1, 0, 0, 0, 0, 0 ],
                          [ 0, 0, 0, 0, 0, 0, 0, 0 ],
                          [ 0, 0, 0, 0, 0, 0, 0, 0 ],
                          [ 0, 0, 0, 0, 0, 0, 0, 0 ],
                          [ 0, 0, 0, 0, 0, 0, 0, 0 ]])

    # plot lines between indicated corner points
    for i in range(0, 8):
        for j in range(0, 8):
            if edge_array[i, j] == 1:
                cv2.line(prism_im, (coords[i, 0], coords[i, 1]), (coords[j, 0], coords[j, 1]), color, 1)
    return prism_im


def draw_base_cross(image, coords, color):
    prism_im = image.copy()
    coords = np.transpose(coords)
    cv2.line(prism_im,
             (((coords[0, 0] + coords[1, 0]) / 2).astype(int), ((coords[0, 1] + coords[1, 1]) / 2).astype(int)),
             (((coords[2, 0] + coords[3, 0]) / 2).astype(int), ((coords[2, 1] + coords[3, 1]) / 2).astype(int)),
             color, 1)
    cv2.line(prism_im,
             (((coords[0, 0] + coords[3, 0]) / 2).astype(int), ((coords[0, 1] + coords[3, 1]) / 2).astype(int)),
             (((coords[1, 0] + coords[2, 0]) / 2).astype(int), ((coords[1, 1] + coords[2, 1]) / 2).astype(int)),
             color, 1)
    return prism_im


def plot_bboxes_3d(image, label, P, style = "normal"):
    """ Plots rectangular prism bboxes on image and returns image
    image - cv2 or PIL style image (function converts to cv2 style image)
    label - for one frame, in the form output by parse_label_file
    P - camera calibration matrix
    bbox_im -  cv2 im with bboxes and labels plotted
    style - string, "ground_truth" or "normal"  ground_truth plots boxes as white
    """
        
    # check type and convert PIL im to cv2 im if necessary
    assert type(image) in [np.ndarray, PIL.PngImagePlugin.PngImageFile], "Invalid image format"
    if type(image) == PIL.PngImagePlugin.PngImageFile:
        image = pil_to_cv(image)
    cv_im = image.copy() 
    
    class_colors = {
            'Cyclist': (255, 150, 0),
            'Pedestrian':(200, 800, 0),
            'Person':(160, 30, 0),
            'Car': (0, 255, 150),
            'Van': (0, 255, 100),
            'Truck': (0, 255, 50),
            'Tram': (0, 100, 255),
            'Misc': (0, 50, 255),
            'DontCare': (200, 200, 200)}
    
    for i in range (0, len(label)):
        if label[i]['pos'][2] > 2 and label[i]['truncation'] < 1:
            cls = label[i]['class']
            if cls != "DontCare":
                bbox_3d, _, _ = get_coords_3d(label[i], P)
                if style == "ground_truth": # for plotting ground truth and predictions
                    cv_im = draw_prism(cv_im,bbox_3d, (255, 255, 255))
                else:
                    cv_im = draw_prism(cv_im,bbox_3d, class_colors[cls])
    return cv_im


def plot_base_3d(image, label, P, style = "normal"):
    """ Plots rectangular prism bboxes on image and returns image
    image - cv2 or PIL style image (function converts to cv2 style image)
    label - for one frame, in the form output by parse_label_file
    P - camera calibration matrix
    bbox_im -  cv2 im with bboxes and labels plotted
    style - string, "ground_truth" or "normal"  ground_truth plots boxes as white
    """
        
    # check type and convert PIL im to cv2 im if necessary
    assert type(image) in [np.ndarray, PIL.PngImagePlugin.PngImageFile], "Invalid image format"
    if type(image) == PIL.PngImagePlugin.PngImageFile:
        image = pil_to_cv(image)
    cv_im = image.copy() 
    
    class_colors = {
            'Cyclist': (255, 150, 0),
            'Pedestrian':(200, 800, 0),
            'Person':(160, 30, 0),
            'Car': (0, 255, 150),
            'Van': (0, 255, 100),
            'Truck': (0, 255, 50),
            'Tram': (0, 100, 255),
            'Misc': (0, 50, 255),
            'DontCare': (200, 200, 200)}
    
    for i in range (0, len(label)):
        if label[i]['pos'][2] > 2 and label[i]['truncation'] < 1:
            cls = label[i]['class']
            if cls != "DontCare":
                bbox_3d, _, _ = get_coords_3d(label[i], P)
                if style == "ground_truth": # for plotting ground truth and predictions
                    cv_im = draw_base(cv_im, bbox_3d, (255, 255, 255))
                    cv_im = draw_base_cross(cv_im, bbox_3d, (255, 0, 0))
                else:
                    cv_im = draw_base(cv_im, bbox_3d, class_colors[cls])
                    cv_im = draw_base_cross(cv_im, bbox_3d, (255, 0, 0))
    return cv_im
