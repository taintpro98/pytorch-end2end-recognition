#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _nms(boxes, threshold=0.5):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of iou
        inter = w * h
        iou = inter / (area[last] + area[idxs[:last]] - inter)
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(iou > threshold)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

def restore_rbox(score_map, geo_map, score_map_thresh = 0.5, nms_thres = 0.5):
    '''
    restore text boxes from score map and geo map
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, :]
    xy_text = np.argwhere(score_map > score_map_thresh)
    xy_text *= 4
    boxes = []
    for c in xy_text:
        c_x, c_y = c[1], c[0]
        top, right, bottom, left = geo_map[c]
        x1 = c_x - left
        y1 = c_y - top
        x2 = c_x + right
        y2 = c_y + bottom
        boxes.append(np.array([x1, y1, x2, y2]))
    boxes = _nms(boxes, nms_thres)
    return boxes

class Converter(object):
    def __init__(self, characters, device):
        # character (str): set of the possible characters.
        dict_characters = list(characters)
        self.device = device
        self.dict = {}
        for i, char in enumerate(dict_characters):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1

        self.characters = ['[blank]'] + dict_characters  # dummy '[blank]' token for CTCLoss (index 0)
        
    def encode(self, texts):
        """
        :param texts: list of texts
        :return: 
            A tensor with shape is total length of all texts
            A tensor representing lengths 
        """
        lengths = [len(s) for s in texts]
        texts = ''.join(texts)
        texts = [self.dict[char] for char in texts]

        return torch.IntTensor(texts).to(self.device), torch.IntTensor(lengths).to(self.device)
    
    def decode(self, text_index, lengths):
        """ convert text-index into text-label. 
            inverse to encode
        :param text_index: A tensor with shape is total length of all texts
        :return: list of texts 
        """
        texts = []
        index = 0
        for l in lengths:
            t = text_index[index:index + l]
            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.characters[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts

