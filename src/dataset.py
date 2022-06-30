#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

from src.environment import accepted_models

no_padd_tokenizers = {
    'Cedille/fr-boris'
}


class DofusDataset(Dataset):
    def __init__(self, corpus: list[str], model_name: str):
        super().__init__()
        self.corpus = corpus

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_name in no_padd_tokenizers:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __len__(self) -> int:
        return len(self.corpus)

    def __getitem__(self, index: int) -> str:
        return self.tokenizer.bos_token + self.corpus[index] + self.tokenizer.eos_token

    def generate_batch(self, batch_sentence: list[str]) -> torch.LongTensor:
        ids = self.tokenizer(batch_sentence, padding=True, return_tensors='pt')
        return ids['input_ids']

    @property
    def vocabulary_size(self):
        return len(self.tokenizer)

    @property
    def eos_token(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_token(self):
        return self.tokenizer.pad_token_id

    @staticmethod
    def load_corpus(filename: str) -> list[str]:
        if filename.endswith('french_books_reviews.csv'):
            columns = ['reader_review']
        elif filename.endswith('data.csv'):
            columns = ['boss_desc', 'rubrikabrax', 'meryde']

        df = pd.read_csv(filename)
        corpus = []
        for col_name in columns:
            values = df[col_name].values
            na = df[col_name].isna()
            corpus.extend(values[~na])
        return corpus

    @staticmethod
    def load_datasets(
        frac_test: float,
        seed: int,
        filename: str,
        model_name: str,
    ) -> tuple:
        corpus = DofusDataset.load_corpus(filename)
        data_train, data_test = train_test_split(
            corpus,
            test_size=frac_test,
            random_state=seed,
            shuffle=True
        )
        return DofusDataset(data_train, model_name), DofusDataset(data_test, model_name)

    @staticmethod
    def load_dataloaders(
        batch_size: int,
        frac_test: float=0.2,
        seed: int=0,
        filename: str='data/data.csv',
        model_name: str='asi/gpt-fr-cased-small',
    ) -> tuple[DataLoader]:
        dataset_train, dataset_test = DofusDataset.load_datasets(
            frac_test,
            seed,
            filename,
            model_name,
        )

        loader_train = DataLoader(
            dataset_train,
            batch_size,
            shuffle=True,
            collate_fn=dataset_train.generate_batch
        )
        loader_test = DataLoader(
            dataset_test,
            batch_size,
            shuffle=False,
            collate_fn=dataset_test.generate_batch
        )

        return loader_train, loader_test

