import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset


class KittiDataset(Dataset):
    def __init__(self, base_root: str, split: str) -> None:
        super().__init__()
        
        base_root = os.path.abspath(base_root)
        assert os.path.isdir(base_root)
        self.base_root = base_root
        
        assert split in ['training', 'validating', 'testing']
        self.split = split
        
        # Calibration Files
        self.calib_dir = os.path.join(base_root, split, 'calib')
        self.calib_files = [os.path.join(self.calib_dir, calib) for calib in os.listdir(self.calib_dir)]
        
        # Image Files
        self.image_dir = os.path.join(base_root, split, 'image_2')
        self.image_files = [os.path.join(self.image_dir, image) for image in os.listdir(self.image_dir)]
        
        # Label Files
        self.label_dir = None
        self.label_files = []
        if (split != 'testing'):
            self.label_dir = os.path.join(base_root, split, 'label_2')
            self.label_files = [os.path.join(self.label_dir, label) for label in os.listdir(self.label_dir)]
        
        self.calib_files.sort()
        self.image_files.sort()
        self.label_files.sort()
            
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, str]]:
        meta = {
            'calib_name': str(os.path.basename(self.calib_files[idx])),
            'image_name': str(os.path.basename(self.image_files[idx])),
            'label_name': str(os.path.basename(self.label_files[idx]))
        }
        calib = self.load_calib(idx)
        image = self.load_image(idx)
        label = self.load_label(idx)
        return calib, image, label, meta
    
    def load_calib(self, idx: int) -> np.ndarray:
        line_list = []
        with open(self.calib_files[idx]) as f:
            for line in f:
                line = line.split()
                line_list.append(line)
        line = line_list[2] # get line corresponding to left color camera
        vals = np.zeros([12])
        for i in range(0, 12):
            vals[i] = float(line[i + 1])
        return vals.reshape((3, 4))
    
    def load_image(self, idx: int) -> np.ndarray:
        image_arr = cv2.imread(self.image_files[idx])
        image_data = cv2.cvtColor(image_arr, code=cv2.COLOR_BGR2RGB)
        
        return image_data
    
    def load_label(self, idx: int) -> List[Dict[str, Any]]:
        if self.split == "testing":
            return None
        line_list = []
        with open(self.label_files[idx]) as f:
            for line in f:
                line = line.split()
                line_list.append(line)
            
        # each line corresponds to one detection
        det_dict_list = []  
        for line in line_list:
            # det_dict holds info on one detection
            det_dict = {}
            det_dict['class']      = str(line[0])
            det_dict['truncation'] = int(float(line[1]))
            det_dict['occlusion']  = int(float(line[2]))
            det_dict['alpha']      = float(line[3]) # obs angle relative to straight in front of camera
            x_min = int(round(float(line[4])))
            y_min = int(round(float(line[5])))
            x_max = int(round(float(line[6])))
            y_max = int(round(float(line[7])))
            det_dict['bbox2d']     = np.array([x_min, y_min, x_max, y_max])
            length = float(line[10])
            width = float(line[9])
            height = float(line[8])
            det_dict['dim'] = np.array([length, width, height])
            x_pos = float(line[11])
            y_pos = float(line[12])
            z_pos = float(line[13])
            det_dict['pos'] = np.array([x_pos, y_pos, z_pos])
            det_dict['rot_y'] = float(line[14])
            det_dict_list.append(det_dict)
        
        return det_dict_list
