#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict

import wandb
import numpy as np
from transformers import AutoModel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

from src.dataset import DofusDataset
from src.model import DofusTransformer


class DofusTrain:
    def __init__(self, config: dict):
        self.__dict__ = config

        self.train_loader, self.test_loader = DofusDataset.load_dataloaders(
            batch_size=self.batch_size,
            frac_test=0.2,
            seed=self.seed,
            model_name=self.model_name,
        )
        dataset = self.train_loader.dataset
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.pad_token)
        self.vocabulary_size = dataset.vocabulary_size

        self.model = DofusTransformer(
            self.model_name,
            self.vocabulary_size,
            dataset.pad_token,
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )

    def summary(self):
        n_ticks = 30
        print(n_ticks * '-', 'Dofus Pretrained Transformer', n_ticks * '-')
        summary(
            self.model,
        )

    def do_batch(self, batch: torch.LongTensor) -> dict[str, torch.FloatTensor]:
        """Do one batch forwarding and compute the loss.

        Input
        -----
            batch: Batch of sentences.
                Shape of [batch_size, sentence_len]

        Output
        ------
            metrics: Dictionnary containing a bunch of metrics.
                All metrics are stored by (name -> value).
                The loss is inside.
        """
        metrics = dict()
        x, y = batch[:, :-1], batch[:, 1:]
        y_pred = self.model(x).view(-1, self.vocabulary_size)
        y = y.reshape(-1)
        metrics['loss'] = self.loss_fn(y_pred, y)

        return metrics

    def eval(self, loader: DataLoader) -> dict:
        self.model.eval()
        metrics = defaultdict(list)

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                m = self.do_batch(batch)
                for name, value in m.items():
                    metrics[name].append(value.item())

        for name, values in metrics.items():
            metrics[name] = np.mean(values)
        return metrics

    def eval_and_log(self):
        metrics = dict()

        train_metrics = self.eval(self.train_loader)
        for name, value in train_metrics.items():
            metrics['Train - ' + name] = value

        eval_metrics = self.eval(self.test_loader)
        for name, value in eval_metrics.items():
            metrics['Eval - ' + name] = value

        wandb.log(metrics)

    def start(self):
        optim = self.optimizer
        self.model.to(self.device)

        with wandb.init(
            project='Dofus Pretrained Transformer',
            entity='pierrotlc',
            group=self.group_wb,
            config=self.__dict__,
        ):
            # Evaluate the perfs first
            self.eval_and_log()

            for e in range(self.n_epochs):
                self.model.train()

                for batch in self.train_loader:
                    optim.zero_grad()
                    batch = batch.to(self.device)

                    metrics = self.do_batch(batch)
                    loss = metrics['loss']
                    loss.backward()
                    optim.step()

                self.eval_and_log()
