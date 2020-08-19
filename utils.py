import torchtext
from torch import Tensor
from torchtext.data import Field
import torch


def field_printer(field: Field, prob_tensor: Tensor, tgt: Tensor) -> Tensor:
    ''' reverse tokenizes tensors using field.vocab, returns [(prob, target), ...]'''
    print(
        list(
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
    )


def get_field_term_weights(field: torchtext.data.field) -> torch.Tensor:
    """ Gets normalized class weights from a torchtext.field object sufficient for passing into cross entropy loss criterion."""
    total_counts = sum(field.vocab.freqs.values())
    inverse_proportion_of_term = {
        k: 1 / (v / total_counts) for k, v in field.vocab.freqs.items()
    }
    return torch.tensor(
        [
            inverse_proportion_of_term[i]
            if i in inverse_proportion_of_term.keys()
            else 1 / total_counts
            for i in field.vocab.itos
        ]
    )
