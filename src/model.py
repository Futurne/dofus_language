#!/usr/bin/env python
# -*- coding: utf-8 -*-

from transformers import GPT2LMHeadModel, GPT2Tokenizer

import torch
import torch.nn as nn

from src.environment import accepted_models


def load_model(model_name: str) -> GPT2LMHeadModel:
    """Load the model and freeze every layers except the head.
    """
    assert model_name in accepted_models, f"Unknown model, please select one from {' '.join(accepted_models)}"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.requires_grad_(True)
    # model.requires_grad_(False)
    # model.lm_head.requires_grad_(True)
    return model

def generate(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, beggining: str, device: str) -> list[str]:
    input_ids = tokenizer.encode(beggining, return_tensors='pt')
    input_ids = input_ids.to(device)

    model.eval()
    model.to(device)

    beam_outputs = model.generate(
        input_ids,
        max_length=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=10
    )
    beam_outputs = tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
    return beam_outputs

