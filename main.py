#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import argparse

import torch

from src.scrap.almanax_scrap import ScrapAlmanax
from src.train import DofusTrain


def prepare_training(config_path: str) -> DofusTrain:
    # Default values, can be overwritten in the yaml file
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dataset_path': 'data/data.csv'
    }

    with open(config_path) as config_file:
        config |= yaml.safe_load(config_file)


    train = DofusTrain(config)
    return train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a pretrained model to speak the Dofus language')
    parser.add_argument('config_file', help='Configuration file (YAML file)')

    args = parser.parse_args()

    train = prepare_training(args.config_file)
    train.summary()
    print('\nContinue?')
    if input('[y/n] > ') != 'y':
        sys.exit(0)

    train.start()

