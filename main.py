#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import argparse
from textwrap import wrap

import torch
import pandas as pd

from src.scrap.almanax_scrap import ScrapAlmanax
from src.train import DofusTrain


def prepare_training(config_path: str) -> DofusTrain:
    # Default values, can be overwritten in the yaml file
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dataset_path': 'data/big_file.txt',
    }

    with open(config_path) as config_file:
        config |= yaml.safe_load(config_file)


    train = DofusTrain(config)
    return train


def preprocess_corpus(corpus: list) -> list:
    """Some preprocessing before saving into the big file.
        - Remove the '\n' in the samples.
        - Cut the samples that are too long.
    """
    # Remove all '\n' in the dataset
    corpus = [
        sample.replace('\n', ' ')
        for sample in corpus
    ]

    MAX_SIZE = 300
    corpus_ = []
    for sample in corpus:
        corpus_.extend(wrap(sample, MAX_SIZE))

    return corpus_


def init_big_file():
    """Initialize the dataset big file by reading the scrapped data.
    """
    corpus = []
    df = pd.read_csv('data/almanax.csv')
    COLUMNS = ['boss_desc', 'rubrikabrax', 'meryde']
    for col_name in COLUMNS:
        values = df[col_name].values
        na = df[col_name].isna()
        corpus.extend(values[~na])

    ENCYCLOPEDIA_FILES = [
        'armes.csv',
        'consommables.csv',
        'familiers.csv',
        'ressources.csv',
        'équipements.csv',
    ]
    for filename in ENCYCLOPEDIA_FILES:
        df = pd.read_csv(f'data/{filename}')
        values = df['description'].values
        na = df['description'].isna()
        corpus.extend(values[~na])

    corpus = preprocess_corpus(corpus)

    with open('data/big_file.txt', 'w') as big_file:
        big_file.write('\n'.join(corpus))  # Every samples are separated by a '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a pretrained model to speak the Dofus language')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', default='config.yaml', help='Configuration file (YAML file)')
    group.add_argument('--init', action='store_true', help='Gather from the datasets all the text information into one big text file')

    args = parser.parse_args()

    if args.init:
        init_big_file()
    elif args.train:
        train = prepare_training(args.train)
        train.summary()
        print('\nContinue?')
        if input('[y/n] > ') != 'y':
            sys.exit(0)

        train.start()

