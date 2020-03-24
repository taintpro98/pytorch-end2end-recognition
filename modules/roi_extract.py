#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
import math
import numpy as np

class RoIExtract(nn.Module):
    def __init__(self, height=8):
        super().__init__()
        self.height = height
    def forward(self, feature_maps, boxes, mapping, device):
        '''
        :param feature_maps(tensor):  B * 32 * 128 * 128
        :param boxes: N * 4
        :param mapping: mapping for image
        :return: N * C(=32)* H(=8) * W 
        '''
        regions = []
        lengths = []
        max_width = 0
        
        for img_index, box in zip(mapping, boxes):
            feature = feature_maps[img_index]
            width = feature.shape[2]
            height = feature.shape[1]
            
            x, y, w, h = box/4 # 512 -> 128
            box_width = math.ceil(self.height * w/h)
            box_width = min(width, box_width) # not to exceed feature map's width
            max_width = box_width if box_width > max_width else max_width
            
            lengths.append(box_width)
            region = feature[:, :self.height, :box_width]
            
            regions.append(region)
        lengths = np.array(lengths)
        
        rois = []
        for region in regions:
            w = region.shape[2]
            channel = region.shape[0]
            if w != max_width:
                padded_part = torch.zeros(channel, self.height, max_width-w).to(device)
                roi = torch.cat([region, padded_part], dim=-1)
            else:
                roi = region
            rois.append(roi)
        rois = torch.stack(rois)
        indices = np.argsort(lengths)
        indices = indices[::-1].copy() # descending order
        lengths = lengths[indices]
        rois = rois[indices]
        
        return rois, lengths, indices




