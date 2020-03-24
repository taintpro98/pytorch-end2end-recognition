#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import DataLoader 
from .dataset import SynthTextDataset
from .datautils import collate_fn
import torch.utils.data as torchdata

class SynthTextDataLoader():
    def __init__(self, config):
        # super(SynthTextDataloader, self).__init__(config)
        dataRoot = config['data_loader']['data_dir']
        self.workers = config['data_loader']['workers']
        self.batch_size = config['data_loader']['batch_size']
        self.train_shuffle = config['data_loader']['shuffle']
        self.val_shuffle = config['validation']['shuffle']
        self.num_workers = config['data_loader']['workers']
        self.batch_idx = 0
        ds = SynthTextDataset(dataRoot)
        self.split = config['validation']['validation_split']

        self.__train_dataset, self.__val_dataset = self.__train_val_split(ds)

    def train(self):
        return DataLoader(self.__train_dataset, num_workers = self.num_workers, batch_size = self.batch_size, shuffle = self.train_shuffle, collate_fn = collate_fn)

    def val(self):
        return DataLoader(self.__val_dataset, num_workers = self.num_workers, batch_size = self.batch_size, shuffle = self.val_shuffle, collate_fn = collate_fn)

    def __train_val_split(self, ds):
        '''

        :param ds: dataset
        :return:
        '''

        try:
            split = float(self.split)
        except:
            raise RuntimeError('Train and val splitting ratio is invalid.')

        val_len = int(split * len(ds))
        train_len = len(ds) - val_len
        train, val = torchdata.random_split(ds, [train_len, val_len])
        return train, val



