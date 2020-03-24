#!/usr/bin/env python
# coding: utf-8

import torch
from tools import *
from base.base_trainer import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self, config, model, loss, train_loader, val_loader, resume_path):
        super(Trainer, self).__init__(config, model, loss, resume_path)

        self.converter = Converter(model.characters, self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.verbosity = config["trainer"]["verbosity"]
        
    def _to_tensor(self, *tensors):
        t = []
        for __tensors in tensors:
            t.append(__tensors.to(self.device))
        return t
    
    def _eval_metrics(self):
        return np.ones(3)
    
    def _train_epoch(self, epoch):
        self.model.train()
        
        total_loss = 0
        total_metrics = 0
        for batch_idx, gt in enumerate(self.train_loader):
            print('Training batch: {}/{}'.format(batch_idx, epoch))
            try:
                image_names, images, score_maps, geo_maps, training_masks, transcripts, boxes, mapping = gt
                images, score_maps, geo_maps, training_masks = self._to_tensor(images, score_maps, geo_maps, training_masks)
            
                self.optimizer.zero_grad()
                pred_score_maps, pred_geo_maps, pred_recogs, pred_boxes, pred_mapping, indices = self.model.forward(images, boxes, mapping)
                transcripts = transcripts[indices]
                pred_boxes = pred_boxes[indices]
                pred_mapping = mapping[indices]
                pred_fns = [image_names[i] for i in pred_mapping]
            
                labels, lengths = self.converter.encode(transcripts)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)
                recogs = (labels, lengths)
            
                cls_loss, geo_loss, rec_loss = self.loss(score_maps, pred_score_maps, geo_maps, pred_geo_maps, recogs, pred_recogs, training_masks)
                loss = cls_loss + geo_loss + rec_loss
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pred_transcripts = []
                if len(pred_mapping) > 0:
                    preds, lengths = pred_recogs
                    _, preds = preds.max(2) # [W, N]
                    for i in range(lengths.numel()):
                        l = lengths[i]
                        p = preds[:l, i]
                        t = self.converter.decode(p, [l])
                        pred_transcripts.append(t)
                    pred_transcripts = np.array(pred_transcripts)

                gt_fns = pred_fns
                total_metrics += self._eval_metrics()
                
                print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} IOU Loss: {:.6f} CLS Loss: {:.6f} Recognition Loss: {:.6f}'.format(
                        epoch,
                        batch_idx * self.train_loader.batch_size,
                        len(self.train_loader) * self.train_loader.batch_size,
                        100.0 * batch_idx / len(self.train_loader),
                        loss.item(), geo_loss.item(), cls_loss.item(), rec_loss.item()))
            except:
                print(image_names)
                raise
        
        
        train_log = {
            'loss': total_loss / len(self.train_loader),
            'precious': total_metrics[0] / len(self.train_loader),
            'recall': total_metrics[1] / len(self.train_loader),
            'hmean': total_metrics[2] / len(self.train_loader)
        }
        return train_log
        
    def _val_epoch(self, epoch):
        self.model.eval()
        
        total_val_metrics = np.zeros(3)
        with torch.no_grad():
            for batch_idx, gt in enumerate(self.val_loader):
                try:
                    image_names, images, score_maps, geo_maps, training_masks, transcripts, boxes, mapping = gt
                    images, score_maps, geo_maps, training_masks = self._to_tensor(images, score_maps, geo_maps, training_masks)
                    pred_score_maps, pred_geo_maps, pred_recogs, pred_boxes, pred_mapping, indices = self.model.forward(images, boxes, mapping)
                    
                    pred_transcripts = []
                    pred_fns = []
                    if len(pred_mapping) > 0:
                        pred_mapping = pred_mapping[indices]
                        pred_boxes = pred_boxes[indices]
                        pred_fns = [image_names[i] for i in pred_mapping]

                        preds, lengths = pred_recogs
                        _, preds = preds.max(2)
                        for i in range(lengths.numel()):
                            l = lengths[i]
                            p = preds[:l, i]
                            t = self.converter.decode(p, [l])
                            pred_transcripts.append(t)
                        pred_transcripts = np.array(pred_transcripts)

                    gt_fns = [image_names[i] for i in mapping]
                    total_val_metrics = self._eval_metrics()
                except:
                    print(image_names)
                    raise
        return {
            'val_precious': total_val_metrics[0] / len(self.val_loader),
            'val_recall': total_val_metrics[1] / len(self.val_loader),
            'val_hmean': total_val_metrics[2] / len(self.val_loader)
        }




