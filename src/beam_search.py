#!/usr/bin/env python
# -*- coding: utf-8 -*-

import einops

from transformers import AutoTokenizer

import torch
import torch.nn as nn
from torch.distributions import Categorical


class BeamSearch:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
    ):
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.eos_token_id
        self.bos_token = tokenizer.bos_token_id
        self.pad_token = tokenizer.pad_token

    @staticmethod
    def sample_from_probs(
        probs: torch.FloatTensor,
        width: int,
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        """Sample from the predicted distributions of the next token.

        Input
        -----
            probs: Distributions of the next token.
                Shape of [batch_size, vocabulary_size].
            width: Number of tokens to sample for each distribution.

        Output
        ------
            sampled: Sampled next tokens.
                Shape of [batch_size, width].
            probs: Probabilities associated with each tokens.
                Shape of [batch_size, width].
        """
        sampled = torch.stack([
            Categorical(probs=p).sample((width, ))
            for p in probs
        ])
        probs = torch.stack([
            torch.index_select(p, dim=0, index=i)
            for p, i in zip(probs, sampled)
        ])

        # probs, predicted = probs[:, -1].topk(k=width, dim=-1)
        return sampled, probs

    @staticmethod
    def indices_terminated(
        tokens: torch.FloatTensor,
        eos_token: int
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        """Split the tokens between the terminated and the non-terminated
        sentence. Return the indices of those two groups.

        Args
        ----
            tokens: The sentences.
                Shape of [batch_size, n_tokens].
            eos_token: Value of the End-of-Sentence token.

        Output
        ------
            terminated: Indices of the terminated sentences (who's got the eos_token).
                Shape of [n_terminated, ].
            non-terminated: Indices of the unfinished sentences.
                Shape of [batch_size-n_terminated, ].
        """
        terminated = [i for i, t in enumerate(tokens) if eos_token in t]
        non_terminated = [i for i, t in enumerate(tokens) if eos_token not in t]
        return torch.LongTensor(terminated), torch.LongTensor(non_terminated)

    @staticmethod
    def append_beams(
        tokens: torch.FloatTensor,
        beams: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Add the beam tokens to the current sentences.
        Duplicate the sentences so one token is added per beam per batch.

        Input
        -----
            tokens: Batch of unfinished sentences.
                Shape of [batch_size, n_tokens].
            beams: Batch of beams for each sentences.
                Shape of [batch_size, n_beams].

        Output
        ------
            tokens: Batch of sentences with one beam per sentence.
                Shape of [batch_size * n_beams, n_tokens+1].
        """
        batch_size, n_beams = beams.shape
        n_tokens = tokens.shape[1]

        tokens = einops.repeat(tokens, 'b t -> b c t', c=n_beams)  # [batch_size, n_beams, n_tokens]
        beams = beams.unsqueeze(dim=2)  # [batch_size, n_beams, 1]

        tokens = torch.cat((tokens, beams), dim=2)  # [batch_size, n_beams, n_tokens+1]
        tokens = tokens.view(batch_size*n_beams, n_tokens+1)  # [batch_size * n_beams, n_tokens+1]
        return tokens

    def search(
        self,
        model: nn.Module,
        beggining: str,
        width: int,
        max_sentences: int,
        max_len: int,
        device: str,
    ) -> list[tuple[str, float]]:
        """Do a beam search to produce probable translations.

        Args
        ----
            model: Autoregressive generative model.
            beggining: Beggining of the sentence to search.
            width: Number of top-k tokens we keep at each stage.
            max_sentences: Maximum number of sentences we keep at the end of each stage.
            max_len: Maximum number of tokens for the sentences.
            device: Device to which we make the inference.

        Output
        ------
            sentences: List of sentences orderer by their likelihood.
        """
        tokens = self.tokenizer(
            self.tokenizer.bos_token + beggining,
            return_tensors='pt'
        )['input_ids']
        tokens_likelihood = torch.FloatTensor([1]).to(device)

        model.to(device)
        tokens = tokens.to(device)

        with torch.no_grad():
            while tokens.shape[1] < max_len:
                batch_size, n_tokens = tokens.shape

                # Get next beams
                predicted = model(tokens)
                probs = torch.softmax(predicted, dim=-1)  # [batch_size, n_tokens, vocabulary_size]
                probs = probs[:, -1]  # Take the predicted probabilities for the next token only
                predicted, probs = BeamSearch.sample_from_probs(probs, width)
                predicted, probs = predicted.to(device), probs.to(device)

                # Separe between terminated sentences and the others
                idx_terminated, idx_not_terminated = BeamSearch.indices_terminated(tokens, self.eos_token)
                idx_terminated, idx_not_terminated = idx_terminated.to(device), idx_not_terminated.to(device)

                terminated = torch.index_select(tokens, dim=0, index=idx_terminated)
                terminated_likelihood = torch.index_select(tokens_likelihood, dim=0, index=idx_terminated)

                filter_t = lambda t: torch.index_select(t, dim=0, index=idx_not_terminated)
                others = filter_t(tokens)
                others_likelihood = filter_t(tokens_likelihood)
                predicted = filter_t(predicted)
                probs = filter_t(probs)

                # Add the top predicted token to the previous tokens
                others = BeamSearch.append_beams(others, predicted)

                # Add padding to terminated tokens
                padd = torch.ones((len(terminated), 1), dtype=torch.long, device=device)
                terminated = torch.cat(
                    (terminated, padd),
                    dim=1
                )

                # Update each tokens likelihood
                others_likelihood = torch.repeat_interleave(others_likelihood, width)
                others_likelihood *= probs.flatten()
                terminated_likelihood *= 0.999  # Penalize short sequences overtime

                # Group up the terminated and the others
                tokens_likelihood = torch.cat(
                    (others_likelihood, terminated_likelihood),
                    dim=0
                )
                tokens = torch.cat(
                    (others, terminated),
                    dim=0
                )

                # Keep only the top `max_sentences` tokens
                if tokens_likelihood.shape[0] > max_sentences:
                    tokens_likelihood, indices = tokens_likelihood.topk(k=max_sentences, dim=0)
                    tokens = torch.index_select(tokens, dim=0, index=indices)


        # Decode tokens to their sentences
        sentences = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        # Join the sentences with their likelihood
        sentences = [(s, p.item()) for s, p in zip(sentences, tokens_likelihood)]
        # Sort the sentences by their likelihood
        sentences = [(s, p) for s, p in sorted(sentences, key=lambda k: k[1], reverse=True)]

        return sentences
