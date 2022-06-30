#!/usr/bin/env python
# -*- coding: utf-8 -*-

from transformers import AutoModel

import torch
import torch.nn as nn

from src.environment import accepted_models


class DofusTransformer(nn.Module):
    def __init__(
        self,
        model_name: str,
        vocabulary_size: int,
        pad_token: int,
        n_layers: int,
        hidden_size: int,
        dropout: float,
    ):
        super().__init__()
        self.pad_token = pad_token

        assert model_name in accepted_models, f"Accepted models: {' '.join(accepted_models)}"
        assert n_layers >= 1

        self.pretrained_model = AutoModel.from_pretrained(model_name)
        self.pretrained_model.requires_grad = False
        for params in self.pretrained_model.parameters():
            params.requires_grad = False

        self.config = self.pretrained_model.config

        with torch.no_grad():
            x = torch.zeros((1, 10), dtype=torch.long)
            x = self.pretrained_model(x)['last_hidden_state']
            output_hidden_size = x.shape[-1]

        self.project_hidden = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(output_hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
        )

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.LeakyReLU(),
            )
            for _ in range(n_layers - 1)
        ])
        
        self.head = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        """Predict the next token autoregressively.

        Input
        -----
            x: Batch of sentences.
                Shape of [batch_size, sentence_len].

        Output
        ------
            y: Next tokens predicted (for each token in a sentence).
                Shape of [batch_size, sentence_len, vocabulary_size].
        """
        mask = x != self.pad_token
        x = self.pretrained_model(x, attention_mask=mask)
        x = x.last_hidden_state

        x = self.project_hidden(x)
        for layer in self.hidden_layers:
            x = layer(x) + x
        y = self.head(x)
        return y

