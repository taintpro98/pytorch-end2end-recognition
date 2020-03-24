#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from tools import *
from modules import feature_extraction
from modules import detector
from modules import recognizer
from modules import roi_extract
import pretrainedmodels as pm

class Model:
    def __init__(self, config, characters):
        self.mode = config['model']['mode']

        bbNet =  pm.__dict__['resnet50'](pretrained='imagenet')

        # bbNet = bbNet.to(torch.device('cuda'))
        self.characters = characters
        n_class = len(characters) + 1

        self.sharedConv = feature_extraction.SharedConv(bbNet, config)
        self.recognizer = recognizer.Recognizer(n_class, 32, 256, config)
        self.detector = detector.Detector(config)
        self.roi_extract = roi_extract.RoIExtract(8)
    
    def parallelize(self):
        self.sharedConv = torch.nn.DataParallel(self.sharedConv)
        self.recognizer = torch.nn.DataParallel(self.recognizer)
        self.detector = torch.nn.DataParallel(self.detector)
        
    def to(self, device):
        self.sharedConv = self.sharedConv.to(device)
        self.detector = self.detector.to(device)
        self.recognizer = self.recognizer.to(device)
        
    def summary(self):
        self.sharedConv.summary()
        self.detector.summary()
        self.recognizer.summary()
    
    def optimize(self, optimizer_type, params):
        optimizer = getattr(optim, optimizer_type)(
            [
                {'params': self.sharedConv.parameters()},
                {'params': self.detector.parameters()},
                {'params': self.recognizer.parameters()},
            ],
            **params
        )
        return optimizer
    
    def train(self):
        self.sharedConv.train()
        self.detector.train()
        self.recognizer.train()
        
    def eval(self):
        self.sharedConv.eval()
        self.detector.eval()
        self.recognizer.eval()
        
    def state_dict(self):
        return {
            '0': self.sharedConv.state_dict(),
            '1': self.detector.state_dict(),
            '2': self.recognizer.state_dict()
        }
    
    def load_state_dict(self, sd):
        self.sharedConv.load_state_dict(sd['0'])
        self.detector.load_state_dict(sd['1'])
        self.recognizer.load_state_dict(sd['2'])
        
    def training(self):
        return self.sharedConv.training and self.detector.training and self.recognizer.training
    
    def forward(self, images, boxes, mapping, is_train=True):
        
        if images.is_cuda:
            device = images.get_device()
        else:
            device = torch.device('cpu')
          
        feature_maps = self.sharedConv.forward(images)
        score_maps, geo_maps = self.detector(feature_maps)
        if is_train:
            rois, lengths, indices = self.roi_extract(feature_maps, boxes[:, :4], mapping, device)
            pred_mapping = mapping
            pred_boxes = boxes
        else:
            scores = score_maps.permute(0, 2, 3, 1)
            geometries = geo_maps.permute(0, 2, 3, 1)
            scores = scores.detach().cpu().numpy()
            geometries = geometries.detach().cpu().numpy()
            
            pred_boxes = []
            pred_mapping = []
            for i in range(scores.shape[0]):
                s = scores[i, :, :, 0]
                g = geometries[i, :, :, ]
                bb = restore_rbox(score_map=s, geo_map=g)
                bb_size = bb.shape[0]

                if len(bb) > 0:
                    pred_mapping.append(np.array([i] * bb_size))
                    pred_boxes.append(bb)
            if len(pred_mapping) > 0:
                pred_boxes = np.concatenate(pred_boxes)
                pred_mapping = np.concatenate(pred_mapping)
                rois, lengths, indices = self.roi_extract(feature_maps, pred_boxes[:, :4], pred_mapping, device)
            else:
                return score_maps, geo_maps, (None, None), pred_boxes, pred_mapping, None
            
        rois = rois.to(device)    
        lengths = torch.tensor(lengths).to(device)

        preds = self.recognizer(rois, lengths) # N, W, nclass
        preds = preds.permute(1, 0, 2) # B, T, C -> T, B, C [W, N, nclass]
        return score_maps, geo_maps, (preds, lengths), pred_boxes, pred_mapping, indices

class FOTSLoss(nn.Module):
    def __init__(self, config):
        super(FOTSLoss, self).__init__()
        self.mode = config["model"]["mode"]
        self.detector_loss = detector.DetectorLoss()
        self.recognizer_loss = recognizer.RecognizerLoss()
        
    def forward(self, y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, y_true_recog, y_pred_recog, training_masks):
        """
        :return:
        """
        cls_loss, geo_loss = self.detector_loss(y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, training_masks)
        rec_loss = self.recognizer_loss(y_true_recog, y_pred_recog)
        return cls_loss, geo_loss, rec_loss




