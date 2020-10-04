"""example code of how we might tokenize individual categorial variables"""

import logging

import numpy as np
import torch
import torchtext

from utils import get_field_term_weights, set_seed

set_seed(0)
logger = logging.getLogger(__name__)
# trn_act_seqs, trn_static_data, tst_act_seqs, tst_static_data = get_dat_data()

from load_staged_acts import TTFieldWrapper

eos_token = "<eos>"
init_token = "<sos>"
pad_token = "<pad>"

from torch import Tensor


class IndependentCategorical(TTFieldWrapper):
    def __init__(
        self,
        name: str,
        num_levels: int,
        padding_idx: int,
        term_weights: Tensor,
        min_frequency: int,
        *args,
        **kwargs
    ) -> None:
        """ independent categorical used to create embedding and classification layers"""
        super().__init__(min_frequency, *args, **kwargs)
        self.name = name
        self.num_levels = num_levels
        self.padding_idx = padding_idx
        self.term_weights = term_weights
        self.min_frequency = min_frequency

    @classmethod
    def from_torchtext_field_wrapper(
        cls,
        name: str,
        field: TTFieldWrapper,
        total_num_samples: int,
        padding_token="<pad>",
    ):
        num_levels: int = len(field.vocab.itos)
        padding_idx: int = field.vocab.stoi[padding_token]  # type: ignore
        term_weights: Tensor = get_field_term_weights(field, total_num_samples)  # type: ignore
        min_frequency: int = field.min_frequency
        return IndependentCategorical(
            name, num_levels, padding_idx, term_weights, min_frequency
        )


TYPE = TTFieldWrapper(
    sequential=False,
    unk_token="<unk>",
    lower=True,
    eos_token=eos_token,
    init_token=init_token,
    pad_token=pad_token,
    min_frequency=100,
)

SUBTYPE = TTFieldWrapper(
    sequential=False,
    unk_token="<unk>",
    lower=True,
    eos_token=eos_token,
    init_token=init_token,
    pad_token=pad_token,
    min_frequency=15,
)

LVL = TTFieldWrapper(
    sequential=False,
    unk_token="<unk>",
    lower=True,
    eos_token=eos_token,
    init_token=init_token,
    pad_token=pad_token,
    min_frequency=50,
)

RESPGROUP = TTFieldWrapper(
    min_frequency=7,
    sequential=False,
    unk_token="<unk>",
    lower=True,
    eos_token=eos_token,
    init_token=init_token,
    pad_token=pad_token,
)


class TTFieldWrapper(torchtext.data.Field):
    def __init__(self, min_frequency: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_frequency = min_frequency


def pad(series_element, pad_len, pad_token, num_fields_to_pad):
    num_pads_needed = pad_len - len(series_element)
    return series_element + [[pad_token] * num_fields_to_pad] * num_pads_needed


a = [[0, 0, 0, 0]]
assert pad(a, 2, 1, 4) == [[0, 0, 0, 0], [1, 1, 1, 1]]


def batchify_act_seqs(data, batch_sz):
    """ build 4d tensor of dims (crs x sequences(max_length) x batchsz x num_act_cats) """
    data = torch.tensor(data)  # type: ignore
    nbatch = data.size(0) // batch_sz
    data = data.narrow(0, 0, nbatch * batch_sz)
    # add a dim to the act cats for future mini batch dim, and then split long ways, along the cr dim
    chunks = torch.chunk(data.unsqueeze(2), batch_sz)  # type: ignore
    # concat along the new act_cats dim to construct the mini batch dim
    return torch.cat(chunks, dim=2)  # type: ignore


def batchify_static_data(static_data, batch_sz):
    n = np.array(static_data)
    n = np.expand_dims(n, 1)
    n = np.concatenate(np.vsplit(n, batch_sz), 1)
    return n


if __name__ == "__main__":
    # process(trn_act_seqs, trn_static_data, tst_act_seqs, tst_static_data)
    pass
