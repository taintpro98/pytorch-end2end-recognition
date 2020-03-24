#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import cv2
import json
import numpy as np
from .datautils import *

# class InvoiceDataset(Dataset):
#     """dataset."""

#     def __init__(self, data_path=DATA_DIR, target_size=768, debug=False):
#         super(InvoiceDataset, self).__init__()
#         self.target_size = target_size
#         self.debug = debug
#         if self.debug:
#             os.makedirs('./debugs', exist_ok=True)
#         self.data_path = data_path
#         images_name = [file for file in os.listdir(os.path.join(data_path, 'images')) if file.endswith('.png') or file.endswith('.jpg')]
#         self.images_name = []
#         for img_path in images_name:
#             if self.check_gt(img_path):
#                 self.images_name.append(img_path)
#             else:
#                 print("[WARNING] Can't found label for {}, skipped".format(img_path))
# #         self.gaussian = GaussianTransformer(imgSize=1024, region_threshold=0.35, affinity_threshold=0.15)
#     def check_gt(self, img_path):
#         path = os.path.join(self.data_path, 'labels', img_path.replace('.png', '.json').replace('.jpg', '.json'))
#         is_file = os.path.isfile(path) 
#         return is_file

#     def __len__(self):
#         return len(self.images_name)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         image_name = self.images_name[idx]
#         image_path = os.path.join(self.data_path, 'images', image_name)
#         raw_image = cv2.imread(image_path)
#         print(idx, raw_image.shape)
#         h, w = raw_image.shape[:2]
#         raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
#         gt_path = os.path.join(self.data_path, 'labels', image_name.replace('.png', '.json').replace('.jpg', '.json'))
#         print(gt_path)
#         raw_anno = json.loads(open(gt_path, 'r').read())
#         regions = self.load_gt(raw_anno)
#         target = {}
#         target["regions"] = regions
#         sample = {"image": raw_image, "target": target}
#         return sample
    
#     def load_gt(self, raw_anno):
#         return raw_anno["attributes"]["_via_img_metadata"]["regions"]
    
    

class SynthTextDataset(Dataset):
    def __init__(self, data_path, input_size=512, debug=False):
        super(SynthTextDataset, self).__init__()
        self.debug = debug
        if self.debug:
            os.makedirs('./debugs', exist_ok=True)
        self.input_size = input_size
        self.data_path = Path(data_path)
        image_names = [file for file in os.listdir(os.path.join(self.data_path, 'images')) if file.endswith('.png') or file.endswith('.jpg')]
        self.image_names = []
        for img_name in image_names:
            if self.check_gt(img_name):
                self.image_names.append(img_name)
            else:
                print("[WARNING] Can't found label for {}, skipped".format(img_name))
       
    def check_gt(self, img_name):
        path = os.path.join(self.data_path, 'labels', img_name.replace('.png', '.json').replace('.jpg', '.json'))
        is_file = os.path.isfile(path) 
        return is_file
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        image_name = self.image_names[idx]
        # gt_path = os.path.join(self.data_path, 'labels', image_name.replace('.png', '.json').replace('.jpg', '.json'))
        gt_path = Path(self.data_path / 'labels' / image_name.replace('.png', '.json').replace('.jpg', '.json'))
        # open(gt_path, 'r', encoding='utf8').read()
        raw_anno = json.loads(open(gt_path, 'r', encoding='utf8').read())
        boxes, transcripts = self.load_gt(raw_anno)
#         try:
        return self.__transform((image_name, boxes, transcripts))
#         except:
#             return self.__getitem__(torch.tensor(np.random.randint(0, len(self))))
      
    def load_gt(self, raw_anno):
        regions = raw_anno["attributes"]["_via_img_metadata"]["regions"]
        boxes = []
        transcripts = []
        for r in regions:
            box = r["shape_attributes"]
            if box["name"] == "rect":
                x, y, w, h = box["x"], box["y"], box["width"], box["height"]
                box = np.array([
                    [x, y],
                    [x+w, y],
                    [x+w, y+h],
                    [x, y+h]
                ])
                boxes.append(box)
                transcript = r["region_attributes"]["label"]
                transcripts.append(transcript)         
        return boxes, transcripts
    
    def __transform(self, gt, input_size = 512, random_scale = np.array([0.5, 1, 2.0, 3.0]), background_ratio = 3. / 8):
        image_name, boxes, transcripts = gt
        image_path = os.path.join(self.data_path, 'images', image_name)
        # print(image_path)
        raw_image = cv2.imread(image_path)
        h, w, _ = raw_image.shape
        
        nwords = len(boxes)
        text_polys = np.stack(boxes, axis=0).astype(np.float32) # nwords * 4 * 2
        text_tags = np.zeros(nwords) # 1 to ignore, 0 to hold
        
        rd_scale = np.random.choice(random_scale)
        img = cv2.resize(raw_image, dsize = None, fx = rd_scale, fy = rd_scale)
        text_polys *= rd_scale
        
        if False:
#             if text_polys.shape[0] > 0:
#                 raise RuntimeError('Cannot find background.')
            # pad and resize the image    
            new_h, new_w, _ = img.shape
            max_h_w_i = np.max([new_h, new_w, input_size])
            im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype = np.uint8)
            im_padded[:new_h, :new_w, :] = img.copy()
            img = cv2.resize(im_padded, dsize = (input_size, input_size))
            score_map = np.zeros((input_size, input_size), dtype = np.uint8)
            
            geo_map_channels = 4
            geo_map = np.zeros((input_size, input_size, geo_map_channels), dtype = np.float32)
            training_mask = np.ones((input_size, input_size), dtype = np.uint8)
        else:
#             if text_polys.shape[0] == 0:
#                 raise RuntimeError('Cannot find background.')
            # pad and resize the image    
            new_h, new_w, _ = img.shape
            max_h_w_i = np.max([new_h, new_w, input_size])
            im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype = np.uint8)
            im_padded[:new_h, :new_w, :] = img.copy()
            img = cv2.resize(im_padded, dsize = (input_size, input_size))
            # resize text polygons
            resize_ratio_3_x = input_size / float(new_w)
            resize_ratio_3_y = input_size / float(new_h)
            text_polys[:, :, 0] *= resize_ratio_3_x
            text_polys[:, :, 1] *= resize_ratio_3_y
            #generate rbox
            score_map, geo_map, training_mask, rectangles = generate_rbox((input_size, input_size), text_polys, text_tags)
            
        image = img[:, :, ::-1].astype(np.float32)  # bgr -> rgb
        score_map = score_map[::4, ::4, np.newaxis].astype(np.float32)
        geo_map = geo_map[::4, ::4, :].astype(np.float32)
        training_mask = training_mask[::4, ::4, np.newaxis].astype(np.float32)
        
        return image_name, image, score_map, geo_map, training_mask, transcripts, rectangles