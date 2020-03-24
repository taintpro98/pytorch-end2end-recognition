#!/usr/bin/env python
# coding: utf-8

import argparse
import os
from data_loader.dataloader import SynthTextDataLoader
import json
from model import Model, FOTSLoss
from trainer import Trainer

def train(config, resume_path):
    data_loader = SynthTextDataLoader(config)
    train_loader = data_loader.train()
    val_loader = data_loader.val()

    characters = json.loads(open(config["dictionary"], encoding='utf8').read())
    model = Model(config, characters)
    model.summary()
    
    loss = FOTSLoss(config)
    
    trainer = Trainer(config, model, loss, train_loader=train_loader, val_loader=val_loader, resume_path=resume_path)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', default='./config.json', help='path to config')
    parser.add_argument('-p', '--resume_path', default='saved_models/', help='path to model weights')
    args = parser.parse_args()
    config = json.load(open(args.config_path))
    resume_path = args.resume_path
    # config = json.loads(open("/Users/macbook/Documents/Cinnamon/E2E/code/rnd_end_2_end/config.json").read())
    # resume_path = 'saved_models/'
    path = os.path.join(config['trainer']['save_dir'], config['name'])
    train(config, resume_path)



