import random

import numpy as np
import torch
import torchtext
from torch import Tensor
from torchtext.data import Field
import logging

logger = logging.getLogger(__name__)


def field_printer(field: Field, prob_tensor: Tensor, tgt: Tensor):
    """ reverse tokenizes tensors using field.vocab, returns [(prob, target), ...]"""

    return list(
        zip(
            [
                field.vocab.itos[i]
                for i in prob_tensor.argmax(dim=-1).flatten()
                if field.vocab.itos[i] != "<pad>"
            ],
            [
                field.vocab.itos[i]
                for i in tgt.flatten()
                if field.vocab.itos[i] != "<pad>"
            ],
        )
    )


def field_accuracy(
    field: Field, prob_tensor: Tensor, tgt: Tensor, digits: int
) -> float:
    acc = [
        1 if i == j else 0
        for i, j in zip(
            [
                field.vocab.itos[i]
                for i in prob_tensor.argmax(dim=-1).flatten()
                if field.vocab.itos[i] != "<pad>"
            ],
            [
                field.vocab.itos[i]
                for i in tgt.flatten()
                if field.vocab.itos[i] != "<pad>"
            ],
        )
    ]

    return round(sum(acc) / len(acc), digits)


def get_field_term_weights(
    field: torchtext.data.field, total_num_samples
) -> torch.Tensor:
    """ Gets normalized class weights from a torchtext.field object sufficient for passing into cross entropy loss criterion."""
    # n_samples / (n_classes * bincount) from scikit learn balanced classes
    inverse_proportion_of_term = {
        class_name: total_num_samples / (len(field.vocab.stoi) * total_counts)
        for class_name, total_counts in field.vocab.freqs.items()
    }

    weights = torch.tensor(
        [
            inverse_proportion_of_term[i] if i in inverse_proportion_of_term.keys()
            # else 1.0
            else total_num_samples / (len(field.vocab.stoi) * total_num_samples)
            for i in field.vocab.itos
        ]
    )
    logger.debug(f"Field:{field}, weights: {weights}")

    return weights


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
