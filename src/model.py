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
    ):
        super().__init__()
        self.pad_token = pad_token

        if model_name not in accepted_models:
            raise RuntimeError(f"Unknown model name: '{model_name}', " +
                               "accepted models: {' '.join(accepted_models)}")

        self.pretrained_model = AutoModel.from_pretrained(model_name)
        self.pretrained_model.requires_grad = False
        for params in self.pretrained_model.parameters():
            params.requires_grad = False

        with torch.no_grad():
            x = torch.zeros((1, 10), dtype=torch.long)
            x = self.pretrained_model(x)['last_hidden_state']
            hidden_size = x.shape[-1]

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, vocabulary_size)
        )

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
        y = self.head(x)
        return y

