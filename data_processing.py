'''example code of how we might tokenize individual categorial variables'''

import pandas
import torchtext
from torchtext.data import Example

from load_staged_acts import get_dat_data

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


def process(trn_act_seqs, trn_static_data, tst_act_seqs, tst_static_data):
    TYPE.build_vocab(
        [
            [act[0] for act in actlist if not pandas.isna(act[0])]
            for actlist in trn_act_seqs
        ]
    )
    SUBTYPE.build_vocab(
        [
            [act[1] for act in actlist if not pandas.isna(act[1])]
            for actlist in trn_act_seqs
        ]
    )
    LVL.build_vocab(
        [
            [act[2] for act in actlist if not pandas.isna(act[2])]
            for actlist in trn_act_seqs
        ]
    )
    RESPGROUP.build_vocab(
        [
            [act[3] for act in actlist if not pandas.isna(act[3])]
            for actlist in trn_act_seqs
        ]
    )

    # add sequencing indicator tokens and numericalize

    def add_start_stop_and_numericalize(act_seq):
        if pandas.isna(act_seq[0][0]):
            return [
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

        else:
            act_seq.insert(0, [init_token] * len(act_seq[0]))
            act_seq.append([eos_token] * len(act_seq[0]))
            return [
                [
                    TYPE.vocab.stoi[act[0]],
                    SUBTYPE.vocab.stoi[act[1]],
                    LVL.vocab.stoi[act[2]],
                    RESPGROUP.vocab.stoi[act[3]],
                ]
                for act in act_seq
            ]

    numer_trn_act_seqs = trn_act_seqs.apply(add_start_stop_and_numericalize)
    sorted_num_seqs = trn_act_seqs.apply(len).sort_values().index
    numer_trn_act_seqs = numer_trn_act_seqs.reindex(sorted_num_seqs)

    numer_tst_act_seqs = tst_act_seqs.apply(add_start_stop_and_numericalize)
    sorted_num_seqs = tst_act_seqs.apply(len).sort_values().index
    numer_tst_act_seqs = numer_tst_act_seqs.reindex(sorted_num_seqs)

    from transformers import LongformerTokenizer, DistilBertTokenizer

    # tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    # numer_trn_static_data = trn_static_data["DESCR"].apply(tokenizer.encode)
    # numer_tst_static_data = tst_static_data["DESCR"].apply(tokenizer.encode)

    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased", return_tensors="pt"
    )
    numer_trn_static_data = trn_static_data["DESCR"].apply(
        lambda x: tokenizer.encode(x[:512])
    )
    numer_tst_static_data = tst_static_data["DESCR"].apply(
        lambda x: tokenizer.encode(x[:512])
    )

    return (
        numer_trn_act_seqs,
        numer_tst_act_seqs,
        numer_trn_static_data,
        numer_tst_static_data,
    )


if __name__ == "__main__":
    # process(trn_act_seqs, trn_static_data, tst_act_seqs, tst_static_data)
    pass
