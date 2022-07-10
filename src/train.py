#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict

import wandb
import einops
import numpy as np
from tqdm import tqdm
from transformers import GPT2LMHeadModel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

from src.dataset import DofusDataset
from src.model import load_model, generate
from src.beam_search import BeamSearch


class DofusTrain:
    def __init__(self, config: dict):
        self.__dict__ = config

        self.train_loader, self.test_loader = DofusDataset.load_dataloaders(
            batch_size=self.batch_size,
            frac_test=0.2,
            seed=self.seed,
            model_name=self.model_name,
            filename=self.dataset_path,
        )
        dataset = self.train_loader.dataset
        self.tokenizer = dataset.tokenizer
        self.vocabulary_size = dataset.vocabulary_size
        self.pad_token = dataset.pad_token
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token)

        self.model = load_model(self.model_name)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.reg_weight,
        )

        self.generator = BeamSearch(dataset.tokenizer)

    def summary(self):
        n_ticks = 30
        print(n_ticks * '-', 'Dofus Pretrained Transformer', n_ticks * '-')
        summary(
            self.model,
        )

        # Datasets summary
        print(f'Training dataset length: {len(self.train_loader.dataset)}')
        print(f'Evaluation dataset length: {len(self.test_loader.dataset)}')

        # General parameters summary
        general_params = [
            'n_epochs',
            'batch_size',
            'lr',
            'reg_weight',
            'group_wb',
            'device',
        ]
        for p in general_params:
            p_exp = f'[{p}]'
            print(f'     {p_exp:<20}-\t\t{self.__dict__[p]}')

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
        y_pred = self.model(x)['logits'].view(-1, self.vocabulary_size)
        y = y.reshape(-1)
        metrics['loss'] = self.loss_fn(y_pred, y)

        probs_predicted = torch.softmax(y_pred, dim=1)
        for k in [1, 3, 10]:
            metrics[f'top-{k} accuracy'] = DofusTrain.topk_accuracy(
                y,
                probs_predicted,
                k,
                self.pad_token
            )

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
        log_table = wandb.Table(columns=['epoch', 'sample'])

        with wandb.init(
            project='Dofus Pretrained Transformer',
            entity='pierrotlc',
            group=self.group_wb,
            config=self.__dict__,
        ):
            # Evaluate the perfs first
            self.eval_and_log()

            for e in tqdm(range(self.n_epochs)):
                self.model.train()

                for batch in tqdm(self.train_loader):
                    optim.zero_grad()
                    batch = batch.to(self.device)

                    metrics = self.do_batch(batch)
                    loss = metrics['loss']
                    loss.backward()
                    optim.step()

                self.eval_and_log()

                sentences = generate(self.model, self.tokenizer, "Aujourd'hui", self.device)
                log_table.add_data(e+1, sentences[0])

            wandb.log({'Samples': log_table})

    @staticmethod
    def topk_accuracy(
        real_tokens: torch.FloatTensor,
        probs_tokens: torch.FloatTensor,
        k: int,
        pad_token: int,
    ) -> torch.FloatTensor:
        """Compute the top-k accuracy.
        We ignore the PAD tokens.

        Args
        ----
            real_tokens: Real tokens of the target sentence.
                Shape of [batch_size * n_tokens].
            probs_tokens: Tokens probability predicted by the model.
                Shape of [batch_size * n_tokens, n_target_vocabulary].
            k: Top-k accuracy threshold.
            pad_token: Padding token value.
        
        Output
        ------
            acc: Scalar top-k accuracy value.
        """
        total = (real_tokens != pad_token).sum()

        _, pred_tokens = probs_tokens.topk(k=k, dim=-1)  # [batch_size * n_tokens, k]
        real_tokens = einops.repeat(real_tokens, 'b -> b k', k=k)  # [batch_size * n_tokens, k]

        good = (pred_tokens == real_tokens) & (real_tokens != pad_token)
        acc = good.sum() / total
        return acc
