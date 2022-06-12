#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset

from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained('Cedille/fr-boris')


class DofusDataset(Dataset):
    def __init__(self, corpus: list[str]):
        super().__init__()
        self.corpus = corpus

    def __len__(self) -> int:
        return len(self.corpus)

    def __getitem__(self, index: int):
        return self.corpus[index]


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('data/data.csv')
    corpus = []
    for col_name in ['boss_desc', 'rubrikabrax', 'meryde']:
        values = df[col_name].values
        corpus.extend(values)

    dataset = DofusDataset(corpus)
    print(dataset[0])
