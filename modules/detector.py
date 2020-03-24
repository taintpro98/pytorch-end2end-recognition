#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from base.base_model import BaseModel 


class Detector(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.scoreLayer = nn.Conv2d(32, 1, kernel_size=1)
        self.geoLayer = nn.Conv2d(32, 4, kernel_size=1)
        
    def forward(self, feature_maps):
        """
        :param feature_maps:
        :return:
            scoreMaps:
            geoMaps:
        """
        
        scoreMaps = self.scoreLayer(feature_maps)
        scoreMaps = torch.sigmoid(scoreMaps)
        
        geoMaps = self.geoLayer(feature_maps)
        geoMaps = self.geoLayer(feature_maps)*512
        
        return scoreMaps, geoMaps

class DetectorLoss(nn.Module):
    def __init__(self):
        super(DetectorLoss, self).__init__()
    
    def forward(self, y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, training_masks):
        """
        :param y_true_cls: B * 1 * 128 * 128 (tensor)
        :param y_pred_cls: B * 1 * 128 * 128 (tensor)
        :param y_true_geo: B * 4 * 128 * 128 (tensor)
        :param y_pred_geo: B * 4 * 128 * 128 (tensor)
        :param training_masks: B * 1 * 128 * 128 (tensor)
        :return:
            : A tensor with shape B * 1 * 128 * 128
            classification_loss: A scalar
        """
        
        classification_loss = self.__dice_coefficient(y_true_cls, y_pred_cls, training_masks)
        classification_loss *= 0.01

        # d1 -> top, d2->right, d3->bottom, d4->left
        d1_gt, d2_gt, d3_gt, d4_gt = torch.split(y_true_geo, 1, 1) # B, 1, 128, 128
        d1_pred, d2_pred, d3_pred, d4_pred = torch.split(y_pred_geo, 1, 1) # B, 1, 128, 128
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt) # B, 1, 128, 128
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred) #B, 1, 128, 128
        w_union = torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred) # B, 1, 128, 128
        h_union = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred) # B, 1, 128, 128
        area_intersect = w_union * h_union # B, 1, 128, 128
        # area_intersect[area_intersect < 0] = 0
        area_union = area_gt + area_pred - area_intersect # B, 1, 128, 128
        L_AABB = -torch.log((area_intersect + 1.0) / (area_union + 1.0)) # B, 1, 128, 128
        return classification_loss, torch.mean(L_AABB * y_true_cls * training_masks)
    
    def __dice_coefficient(self, y_true_cls, y_pred_cls,
                         training_masks):
        """
        dice loss
        :param y_true_cls: B * 1 * 128 * 128 (tensor)
        :param y_pred_cls: B * 1 * 128 * 128 (tensor)
        :param training_masks: B * 1 * 128 * 128 (tensor)
        :return: A tensor with shape B * 1 * 128 * 128
        """
        eps = 1e-5
        intersection = torch.sum(y_true_cls * y_pred_cls * training_masks)
        union = torch.sum(y_true_cls * training_masks) + torch.sum(y_pred_cls * training_masks) + eps
        loss = 1. - (2 * intersection / union)

        return loss 
    
