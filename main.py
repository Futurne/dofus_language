#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import yaml

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
    if len(sys.argv) < 2 or sys.argv[1] not in {'init', 'train'}:
        print(f'Usage: python3 {sys.argv[0]} [init/train]')
        sys.exit(0)

    if sys.argv[1] == 'init':
        print(f'Downloading data...')
        almanax = ScrapAlmanax()
        almanax.scrap()

        if not os.path.isdir('data'):
            os.makedirs('data')
        almanax.to_csv('data/data.csv')

        print(f'Done !')
        sys.exit(0)

    if sys.argv[1] == 'train':
        train = prepare_training(sys.argv[2])
        train.summary()
        train.start()

