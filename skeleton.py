""" toy example of next sequence prediction at the intra-sequential element level using 
transfomer with sequence level static data

ie. ['sos','sos'] -> ['t1', 's1'] where t and s are from different categorical sets, 
but where value of  s1 is dependent on t1
"""

from typing import List
import numpy as np
import pandas
import torch
import torch.nn as nn

from act_mod.data_processing import (
    LVL,
    RESPGROUP,
    SUBTYPE,
    TYPE,
    process,
    pad_series_to_max_len,
    truncate_series_by_len,
)
from act_mod.load_staged_acts import get_dat_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data is 4 batches of bptt 2, minibatch size 2,  3 act categories
data = [
    [
        [["sos", "sos", "sos"], ["sos", "sos", "sos"]],
        [["tx0", "stx0", "lx0"], ["ty0", "sty0", "ly0"]],
    ],
    [  # bptt record 1
        [["tx0", "stx0", "lx0"], ["ty0", "sty0", "ly0"]],
        [["tx1", "stx1", "lx1"], ["ty1", "sty1", "ly1"]],
    ],
    [  # bptt record 2
        [["tx1", "stx1", "lx1"], ["ty1", "sty1", "ly1"]],
        [["eos", "eos", "eos"], ["ty2", "sty2", "ly2"]],
    ],
    [  # bptt record N
        [["eos", "eos", "eos"], ["ty2", "sty2", "ly2"]],
        [["eos", "eos", "eos"], ["eos", "eos", "eos"]],
    ],
]

trnseq, tstseq, trnstat, tststat = get_dat_data()
(
    numer_trn_act_seqs,
    numer_tst_act_seqs,
    numer_trn_static_data,
    numer_tst_static_data,
) = process(trnseq, trnstat, tstseq, tststat)

assert len(numer_trn_act_seqs) == len(numer_trn_static_data)
assert numer_trn_act_seqs.index[0] == numer_trn_static_data.index[0]

max_len = 21
num_act_cats = 4
batch_sz = 2
trunc_numer_trn_act_seqs = truncate_series_by_len(numer_trn_act_seqs, max_len)
padded_numer_trn_act_seqs = pad_series_to_max_len(
    trunc_numer_trn_act_seqs, pad_token=2, pad_len=max_len
)


def batchify_act_seqs(data, batch_sz):
    """ build 4d tensor of dims (crs x sequences(max_length) x batchsz x num_act_cats) """
    data = torch.tensor(data)
    nbatch = data.size(0) // batch_sz
    data = data.narrow(0, 0, nbatch * batch_sz)
    # add a dim to the act cats for future mini batch dim, and then split long ways, along the cr dim
    chunks = torch.chunk(data.unsqueeze(2), batch_sz)
    # concat along the new act_cats dim to construct the mini batch dim
    return torch.cat(chunks, dim=2).to(device)


seq_data_trn = batchify_act_seqs(padded_numer_trn_act_seqs, batch_sz).to(device)


def batchify_static_data(static_data, batch_sz):
    n = np.array(static_data)
    n = np.expand_dims(n, 1)
    n = np.concatenate(np.vsplit(n, batch_sz), 1)
    return n


static_data_trn = batchify_static_data(numer_trn_static_data, batch_sz)


def gen_inp_data_set(seq_data: torch.Tensor, static_data: np.array):
    """generator that advances through the crs dimension, 
    one cr at a time, generating sequence input and target sets and static data"""
    for i in range(len(seq_data)):
        inp = seq_data[i, 0:-1]
        target = seq_data[i, 1:]
        yield inp, target, static_data[i]


# toy static data, unique token sequences
# this should help model seperate the sos -> tx0/ty0 sequence prediction
static_data = [[1, 2, 3], [4, 5, 6]]

num_type_tokens = len(TYPE.vocab.itos)
num_subtype_tokens = len(SUBTYPE.vocab.itos)
num_lvl_tokens = len(LVL.vocab.itos)
num_rspgrp_tokens = len(RESPGROUP.vocab.itos)

emb_dim = 100
num_seqences_into_tran = 300
num_attn_heads = 2
num_dec_layers = 2
# dims (mini_batch(batch_sz) x bptt x act_cats)
bptt = 2  # num lagged activities
batch_sz = 2  # num crs in mini batch

te = nn.Embedding(num_type_tokens, emb_dim).to(device)
ste = nn.Embedding(num_subtype_tokens, emb_dim).to(device)
le = nn.Embedding(num_lvl_tokens, emb_dim).to(device)
rspgrpe = nn.Embedding(num_rspgrp_tokens, emb_dim).to(device)

tfmr_dec_l = nn.TransformerDecoderLayer(num_seqences_into_tran, num_attn_heads).to(
    device
)
tfmr_dec = nn.TransformerDecoder(tfmr_dec_l, num_dec_layers).to(device)
tfmr_enc_l = nn.TransformerEncoderLayer(768, 1).to(device)
tfmr_enc = nn.TransformerEncoder(tfmr_enc_l, 1).to(device)
tc = nn.Linear(num_seqences_into_tran, num_type_tokens).to(device)
stc = nn.Linear(num_seqences_into_tran, num_subtype_tokens).to(device)
ltc = nn.Linear(num_seqences_into_tran, num_lvl_tokens).to(device)
rspgrpc = nn.Linear(num_seqences_into_tran, num_rspgrp_tokens).to(device)

crit = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(
    [
        {"params": tc.parameters()},
        {"params": stc.parameters()},
        {"params": ltc.parameters()},
        {"params": tfmr_dec.parameters()},
        {"params": te.parameters()},
        {"params": ste.parameters()},
        {"params": le.parameters()},
        {"params": tfmr_enc.parameters()},
    ]
)

from transformers import DistilBertModel, DistilBertTokenizer

static_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased", return_tensors="pt"
)

tfmr_dec.train().to(device)

# make 4d tensor of (bpttbatch x minibatch x activities lookback window(seq lengths) x activity categories)
data_ten = torch.tensor(data_array).long().to(device)

# simple static data tensor
st_data_ten = torch.tensor(static_data).long().to(device)

i = 0
epochs = 15  # 30 seems enough to memorize this toy set
# TODO: static data does not seem to make a difference in model differentiating tx0/ty0, and about 40 epochs loss plataeus
for i in range(epochs):

    print(i)
    tgt_loss = 0.0
    optimizer.zero_grad()

    data_gen = gen_inp_data_set(seq_data_trn, static_data_trn)

    # data, target_data, static_data = next(data_gen)
    for data, tgt, static_data in data_gen:

        static_tok_output = tokenizer(
            static_data.tolist(), padding=True, truncation=True, return_tensors="pt"
        )
        em_st_src = static_model(**static_tok_output)[0].mean(1).unsqueeze(0).to(device)

        emb_type = te(data[..., 0])
        emb_subtype = ste(data[..., 1])
        emb_lvl = le(data[..., 2])
        emb_rspgrp = rspgrpe(data[..., 3])

        # em_st_src = se(st_data_ten).flatten(1).unsqueeze(0).repeat(1, 1, 1)
        dt_src = torch.cat([emb_type, emb_subtype, emb_lvl, emb_rspgrp], dim=2)

        # concat activity category vectors together in last dimension
        # dt_src = torch.reshape(embedded_src, (bptt, batch_sz, src.shape[-1] * emb_dim))

        # mask any future sequences so attention will not use them
        mask = (torch.triu(torch.ones(20, 20)) == 1).transpose(0, 1).to(device)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        ).to(device)

        # process static data
        tfmr_enc_out = tfmr_enc.forward(em_st_src)

        # forward pass main transfomer
        tfmr_out = tfmr_dec.forward(dt_src, memory=em_st_src, tgt_mask=mask)

        # get class probs of activity elements
        tclsprb = tc.forward(tfmr_out)
        stclsprb = stc.forward(tfmr_out)
        lclsprb = ltc.forward(tfmr_out)
        rspgrpsprb = rspgrpc.forward(tfmr_out)

        tclsprb = tclsprb.reshape(4, num_type_tokens)
        stclsprb = stclsprb.reshape(4, num_subtype_tokens)
        lclsprb = lclsprb.reshape(4, num_lvl_tokens)
        lclsprb = rspgrpsprb.reshape(4, num_rspgrp_tokens)

        tgt_loss += crit(tclsprb, tgt[..., 0].flatten())
        tgt_loss += crit(stclsprb, tgt[..., 1].flatten())
        tgt_loss += crit(lclsprb, tgt[..., 2].flatten())
        tgt_loss += crit(lclsprb, tgt[..., 2].flatten())

        tgt_loss.backward()
        optimizer.step()
        print(tgt_loss)

        tgt_loss = 0.0

    if i % 10 == 0:
        print(tclsprb.argmax(dim=-1), stclsprb.argmax(dim=-1), lclsprb.argmax(dim=-1))

# turn on eval
tfmr_dec.eval().to(device)
tfmr_enc.eval().to(device)
te.eval().to(device)
ste.eval().to(device)
le.eval().to(device)
e.eval().to(device)
ltc.eval().to(device)
stc.eval().to(device)
tc.eval().to(device)

# "validate" we have memorized data acceptably
with torch.no_grad():
    k = 0
    src = data_ten[0, :, :, :][0]
    tgt = data_ten[0, :, :, :][1]
    embedded_src = e(src)
    dt_src = torch.reshape(embedded_src, (1, 2, 300))
    tfmr_out = tfmr_dec(dt_src, memory=tfmr_enc_out)

    tclsprb = tc.forward(tfmr_out)
    stclsprb = stc.forward(tfmr_out)
    lclsprb = ltc.forward(tfmr_out)

    tclsprb = tclsprb.reshape(2, n_tokens)
    stclsprb = stclsprb.reshape(2, n_tokens)
    lclsprb = lclsprb.reshape(2, n_tokens)

    tgt_loss = crit(tclsprb, tgt[..., 0].flatten())
    tgt_loss += crit(stclsprb, tgt[..., 1].flatten())
    tgt_loss += crit(lclsprb, tgt[..., 2].flatten())

    print(tgt_loss)
    print(tclsprb.argmax(dim=-1), stclsprb.argmax(dim=-1), lclsprb.argmax(dim=-1))
    print(tgt)

