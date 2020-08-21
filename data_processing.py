"""example code of how we might tokenize individual categorial variables"""

import pandas
import torchtext
from pandas import Series
import torch
import numpy as np

# trn_act_seqs, trn_static_data, tst_act_seqs, tst_static_data = get_dat_data()

eos_token = "<eos>"
init_token = "<sos>"
pad_token = "<pad>"

TYPE = torchtext.data.Field(
    sequential=False,
    unk_token="<unk>",
    lower=True,
    eos_token=eos_token,
    init_token=init_token,
    pad_token=pad_token,
)

SUBTYPE = torchtext.data.Field(
    sequential=False,
    unk_token="<unk>",
    lower=True,
    eos_token=eos_token,
    init_token=init_token,
    pad_token=pad_token,
)

LVL = torchtext.data.Field(
    sequential=False,
    unk_token="<unk>",
    lower=True,
    eos_token=eos_token,
    init_token=init_token,
    pad_token=pad_token,
)

RESPGROUP = torchtext.data.Field(
    sequential=False,
    unk_token="<unk>",
    lower=True,
    eos_token=eos_token,
    init_token=init_token,
    pad_token=pad_token,
)


def truncate_series_by_len(series: Series, max_len: int):
    return series[series.apply(len) <= max_len]


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
    """ Main data processing function that will build independent category tokenizers, 
    numericalize the tokens, and truncate and pad to maximum sequence length."""
    TYPE.build_vocab(
        [
            [act[0] for act in actlist if not pandas.isna(act[0])]
            for actlist in trn_act_seqs
        ],
        specials=["<pad>"],
    )
    SUBTYPE.build_vocab(
        [
            [act[1] for act in actlist if not pandas.isna(act[1])]
            for actlist in trn_act_seqs
        ],
        specials=["<pad>"],
    )
    LVL.build_vocab(
        [
            [act[2] for act in actlist if not pandas.isna(act[2])]
            for actlist in trn_act_seqs
        ],
        specials=["<pad>"],
    )
    RESPGROUP.build_vocab(
        [
            [act[3] for act in actlist if not pandas.isna(act[3])]
            for actlist in trn_act_seqs
        ],
        specials=["<pad>"],
    )

    # add sequencing indicator tokens and numericalize
    trn_act_seqs = truncate_series_by_len(
        trn_act_seqs, max_len - 2
    )  # leave 2 spaces for sos and eos tokens
    tst_act_seqs = truncate_series_by_len(tst_act_seqs, max_len - 2)

    numer_trn_act_seqs = trn_act_seqs.apply(
        add_start_stop_and_numericalize_and_pad, args=(max_len,)
    )
    shuffled_num_seqs_trn = trn_act_seqs.apply(len).sample(frac=1).index
    numer_trn_act_seqs = numer_trn_act_seqs.reindex(shuffled_num_seqs_trn)

    numer_tst_act_seqs = tst_act_seqs.apply(
        add_start_stop_and_numericalize_and_pad, args=(max_len,)
    )
    shuffled_num_seqs_tst = tst_act_seqs.apply(len).sample(frac=1).index
    numer_tst_act_seqs = numer_tst_act_seqs.reindex(shuffled_num_seqs_tst)

    numer_trn_static_data = trn_static_data["TEXT"].fillna("<unk>")  # .apply(
    # lambda x: tokenizer.encode(x[:512])
    # )
    numer_tst_static_data = tst_static_data["TEXT"].fillna("<unk>")  # .apply(
    # lambda x: tokenizer.encode(x[:512])
    # )

    numer_trn_static_data = numer_trn_static_data.reindex(shuffled_num_seqs_trn)
    numer_tst_static_data = numer_tst_static_data.reindex(shuffled_num_seqs_tst)

    return (
        numer_trn_act_seqs,
        numer_tst_act_seqs,
        numer_trn_static_data,
        numer_tst_static_data,
    )


if __name__ == "__main__":
    # process(trn_act_seqs, trn_static_data, tst_act_seqs, tst_static_data)
    pass
