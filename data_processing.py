"""example code of how we might tokenize individual categorial variables"""

import pandas
import torchtext
from pandas import Series
import torch
import numpy as np
from utils import set_seed
import logging

set_seed(0)
logger = logging.getLogger(__name__)
# trn_act_seqs, trn_static_data, tst_act_seqs, tst_static_data = get_dat_data()

from load_staged_acts import TTFieldWrapper

eos_token = "<eos>"
init_token = "<sos>"
pad_token = "<pad>"

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


def pad_series_to_max_len(series: pandas.Series, pad_token, pad_len):
    num_fields_to_pad = (
        len(series[0][0]) if not series.empty and len(series[0]) >= 1 else 1
    )
    return series.apply(lambda x: pad(x, pad_len, pad_token, num_fields_to_pad))


def batchify_act_seqs(data, batch_sz):
    """ build 4d tensor of dims (crs x sequences(max_length) x batchsz x num_act_cats) """
    data = torch.tensor(data)
    nbatch = data.size(0) // batch_sz
    data = data.narrow(0, 0, nbatch * batch_sz)
    # add a dim to the act cats for future mini batch dim, and then split long ways, along the cr dim
    chunks = torch.chunk(data.unsqueeze(2), batch_sz)
    # concat along the new act_cats dim to construct the mini batch dim
    return torch.cat(chunks, dim=2)


def batchify_static_data(static_data, batch_sz):
    n = np.array(static_data)
    n = np.expand_dims(n, 1)
    n = np.concatenate(np.vsplit(n, batch_sz), 1)
    return n


def numericalize(act_seq, *args):
    TYPE = args[0]
    SUBTYPE = args[1]
    LVL = args[2]
    RESPGROUP = args[3]

    return [
        [
            TYPE.vocab.stoi[act[0]],
            SUBTYPE.vocab.stoi[act[1]],
            LVL.vocab.stoi[act[2]],
            RESPGROUP.vocab.stoi[act[3]],
        ]
        for act in act_seq
    ]


def add_start_stop_and_numericalize_and_pad(act_seq, max_len):
    if pandas.isna(act_seq[0][0]):
        act_seq = [
            [
                TYPE.vocab.stoi[init_token],
                SUBTYPE.vocab.stoi[init_token],
                LVL.vocab.stoi[init_token],
                RESPGROUP.vocab.stoi[init_token],
            ],
            [
                TYPE.vocab.stoi[eos_token],
                SUBTYPE.vocab.stoi[eos_token],
                LVL.vocab.stoi[eos_token],
                RESPGROUP.vocab.stoi[eos_token],
            ],
        ]
        # assume type, st, lvl, rg have pad tokens that reference the same numericalized value
        act_seq.extend([[TYPE.vocab.stoi[pad_token]] * 4] * (max_len - len(act_seq)))
        return act_seq

    else:
        act_seq.insert(0, [init_token] * len(act_seq[0]))
        act_seq.append([eos_token] * len(act_seq[0]))
        act_seq.extend([[pad_token] * 4] * (max_len - len(act_seq)))
        return numericalize(act_seq, TYPE, SUBTYPE, LVL, RESPGROUP)


def process(
    trn_act_seqs: Series,
    trn_static_data: Series,
    tst_act_seqs: Series,
    tst_static_data: Series,
    max_len: int,
):
    """Main data processing function that will build independent category tokenizers,
    numericalize the tokens, and truncate and pad to maximum sequence length."""

    logger.info(f"Training Records: {len(trn_act_seqs)}")

    logger.info(f"Test Records: {len(tst_act_seqs)}")

    TYPE.build_vocab(
        [
            [act[0] for act in actlist if not pandas.isna(act[0])]
            for actlist in trn_act_seqs
        ],
        specials=["<pad>"],
        min_freq=100,
    )
    SUBTYPE.build_vocab(
        [
            [act[1] for act in actlist if not pandas.isna(act[1])]
            for actlist in trn_act_seqs
        ],
        specials=["<pad>"],
        min_freq=15,
    )
    LVL.build_vocab(
        [
            [act[2] for act in actlist if not pandas.isna(act[2])]
            for actlist in trn_act_seqs
        ],
        specials=["<pad>"],
        min_freq=50,
    )
    RESPGROUP.build_vocab(
        [
            [act[3] for act in actlist if not pandas.isna(act[3])]
            for actlist in trn_act_seqs
        ],
        specials=["<pad>"],
        min_freq=7,
    )

    # add sequencing indicator tokens and numericalize
    # leave 2 spaces for sos and eos tokens during truncation
    trn_act_seqs = trn_act_seqs.apply(lambda x: x[: max_len - 2])
    tst_act_seqs = tst_act_seqs.apply(lambda x: x[: max_len - 2])

    numer_trn_act_seqs = trn_act_seqs.apply(
        add_start_stop_and_numericalize_and_pad, args=(max_len,)
    )

    numer_tst_act_seqs = tst_act_seqs.apply(
        add_start_stop_and_numericalize_and_pad, args=(max_len,)
    )

    numer_trn_static_data = trn_static_data["TEXT"]  # .fillna("<unk>")  # .apply(
    # lambda x: tokenizer.encode(x[:512])
    # )
    numer_tst_static_data = tst_static_data["TEXT"]  # .fillna("<unk>")  # .apply(
    # lambda x: tokenizer.encode(x[:512])
    # )

    assert numer_trn_act_seqs.apply(len).max() <= max_len
    assert numer_tst_act_seqs.apply(len).max() <= max_len
    assert numer_trn_act_seqs.index[100] == numer_trn_static_data.index[100]
    assert numer_tst_act_seqs.index[100] == numer_tst_act_seqs.index[100]

    return (
        numer_trn_act_seqs,
        numer_tst_act_seqs,
        numer_trn_static_data,
        numer_tst_static_data,
    )


if __name__ == "__main__":
    # process(trn_act_seqs, trn_static_data, tst_act_seqs, tst_static_data)
    pass
