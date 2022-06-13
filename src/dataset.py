#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader


# tokenizer = AutoTokenizer.from_pretrained('Cedille/fr-boris')
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # If using Cedille
tokenizer = AutoTokenizer.from_pretrained('asi/gpt-fr-cased-small')


class DofusDataset(Dataset):
    def __init__(self, corpus: list[str]):
        super().__init__()
        self.corpus = corpus

    def __len__(self) -> int:
        return len(self.corpus)

    def __getitem__(self, index: int) -> str:
        return self.corpus[index] + tokenizer.eos_token

    @staticmethod
    def load_corpus(filename: str) -> list[str]:
        df = pd.read_csv(filename)
        corpus = []
        for col_name in ['boss_desc', 'rubrikabrax', 'meryde']:
            values = df[col_name].values
            corpus.extend(values)
        return corpus

    @staticmethod
    def load_datasets(
        frac_test: float,
        seed: int,
        filename: str
    ) -> tuple:
        corpus = DofusDataset.load_corpus(filename)
        data_train, data_test = train_test_split(
            corpus,
            test_size=frac_test,
            random_state=seed,
            shuffle=True
        )
        return DofusDataset(data_train), DofusDataset(data_test)

    @staticmethod
    def generate_batch(batch_sentence: list[str]) -> torch.LongTensor:
        ids = tokenizer(batch_sentence, padding=True, return_tensors='pt')
        return ids['input_ids']

    @staticmethod
    def load_dataloaders(
        batch_size: int,
        frac_test: float=0.2,
        seed: int=0,
        filename: str='data/data.csv'
    ) -> tuple[DataLoader]:
        dataset_train, dataset_test = DofusDataset.load_datasets(
            frac_test,
            seed,
            filename
        )

        loader_train = DataLoader(
            dataset_train,
            batch_size,
            shuffle=True,
            collate_fn=DofusDataset.generate_batch
        )
        loader_test = DataLoader(
            dataset_test,
            batch_size,
            shuffle=False,
            collate_fn=DofusDataset.generate_batch
        )

        return loader_train, loader_test


if __name__ == '__main__':
    data_train, data_test = DofusDataset.load_dataloaders(64, filename='../data/data.csv')
    print(next(iter(data_train)).shape)
