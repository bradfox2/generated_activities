""" toy example of next sequence prediction at the intra-sequential element level using 
transfomer with sequence level static data

ie. ['sos','sos'] -> ['t1', 's1'] where t and s are from different categorical sets, 
but where value of  s1 is dependent on t1
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW, DistilBertModel, DistilBertTokenizer

from data_processing import (
    LVL,
    RESPGROUP,
    SUBTYPE,
    TYPE,
    pad_series_to_max_len,
    process,
    truncate_series_by_len,
)
from load_staged_acts import get_dat_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example data is 4 batches of bptt 2, minibatch size 2,  3 act categories
# data = [
#     [
#         [["sos", "sos", "sos"], ["sos", "sos", "sos"]],
#         [["tx0", "stx0", "lx0"], ["ty0", "sty0", "ly0"]],
#     ],
#     [  # bptt record 1
#         [["tx0", "stx0", "lx0"], ["ty0", "sty0", "ly0"]],
#         [["tx1", "stx1", "lx1"], ["ty1", "sty1", "ly1"]],
#     ],
#     [  # bptt record 2
#         [["tx1", "stx1", "lx1"], ["ty1", "sty1", "ly1"]],
#         [["eos", "eos", "eos"], ["ty2", "sty2", "ly2"]],
#     ],
#     [  # bptt record N
#         [["eos", "eos", "eos"], ["ty2", "sty2", "ly2"]],
#         [["eos", "eos", "eos"], ["eos", "eos", "eos"]],
#     ],
# ]

trnseq, tstseq, trnstat, tststat = get_dat_data()

 # length of sequence of staged activities, first and last will be sos and eos tokens, resepectively.  any remaining staged activity spaces will be padded with <unk>
sequence_length = 5

(
    numer_trn_act_seqs,
    numer_tst_act_seqs,
    numer_trn_static_data,
    numer_tst_static_data,
) = process(trnseq, trnstat, tstseq, tststat, sequence_length)

assert len(numer_trn_act_seqs) == len(numer_trn_static_data)
assert numer_trn_act_seqs.index[0] == numer_trn_static_data.index[0]

# add one to account for target shifting
max_len = sequence_length + 1
num_act_cats = 4
batch_sz = 8

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


static_data_trn = batchify_static_data(
    numer_trn_static_data[: seq_data_trn.shape[0] * batch_sz], batch_sz
)


def gen_inp_data_set(seq_data: torch.Tensor, static_data: np.array):
    """generator that advances through the crs dimension, 
    one cr at a time, generating sequence input and target sets and static data"""
    for i in range(len(seq_data)):
        inp = seq_data[i, 0:-1]
        target = seq_data[i, 1:]
        yield inp, target, static_data[i]


num_type_tokens = len(TYPE.vocab.itos)
num_subtype_tokens = len(SUBTYPE.vocab.itos)
num_lvl_tokens = len(LVL.vocab.itos)
num_rspgrp_tokens = len(RESPGROUP.vocab.itos)

emb_dim = 100
embedding_dim_into_tran = emb_dim * num_act_cats
num_attn_heads = 2
num_dec_layers = 2
# dims (mini_batch(batch_sz) x bptt x act_cats)
bptt = sequence_length  # sequence of activities of cr

te = nn.Embedding(num_type_tokens, emb_dim).to(device)
ste = nn.Embedding(num_subtype_tokens, emb_dim).to(device)
le = nn.Embedding(num_lvl_tokens, emb_dim).to(device)
rspgrpe = nn.Embedding(num_rspgrp_tokens, emb_dim).to(device)

act_emb_layer_norm = nn.LayerNorm(emb_dim * num_act_cats).to(device)

tfmr_dec_l = nn.TransformerDecoderLayer(embedding_dim_into_tran, num_attn_heads).to(
    device
)
tfmr_dec = nn.TransformerDecoder(tfmr_dec_l, num_dec_layers).to(device)
tfmr_enc_l = nn.TransformerEncoderLayer(400, 1).to(device)
tfmr_enc = nn.TransformerEncoder(tfmr_enc_l, 1).to(device)
tc = nn.Linear(embedding_dim_into_tran, num_type_tokens).to(device)
stc = nn.Linear(embedding_dim_into_tran, num_subtype_tokens).to(device)
ltc = nn.Linear(embedding_dim_into_tran, num_lvl_tokens).to(device)
rspgrpc = nn.Linear(embedding_dim_into_tran, num_rspgrp_tokens).to(device)

tmfr_out_layer_norm = nn.LayerNorm(400).to(device)

bertsqueeze = nn.Linear(768, 400).to(device)
static_model = DistilBertModel.from_pretrained("./distilbert_weights/").to(device)
tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased", return_tensors="pt"
)
bert_squeeze_layer_norm = nn.LayerNorm(400).to(device)


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
        {"params": act_emb_layer_norm.parameters()},
        {"params": tfmr_enc.parameters()},
        {"params": rspgrpe.parameters()},
        {"params": rspgrpc.parameters()},
        {"params": bert_squeeze_layer_norm.parameters()},
        {"params": bertsqueeze.parameters()},
        # {"params": tmfr_out_layer_norm.parameters()},
    ],
    lr=0.1,
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5.0, gamma=0.95)


db_optim = AdamW(static_model.parameters(), lr=1e-5)

tfmr_dec.train().to(device)
tfmr_enc.train().to(device)
bert_squeeze_layer_norm.train()
act_emb_layer_norm.train()
static_model.train()

i = 0
epochs = 15

update_increment = 1
log_interval = 200
for i in range(epochs):

    print(i)
    total_loss = 0.0
    data_gen = gen_inp_data_set(seq_data_trn, static_data_trn)

    counter = 0
    #data, tgt, static_data = next(data_gen)
    for data, tgt, static_data in tqdm(
        data_gen, total=len(seq_data_trn), position=0, leave=True
    ):

        static_tok_output = tokenizer(
            static_data.tolist(), padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        em_st_src = static_model(**static_tok_output)[0].mean(1).unsqueeze(0)
        em_st_src = bertsqueeze(em_st_src)
        em_st_src = bert_squeeze_layer_norm(em_st_src)

        emb_type = te(data[..., 0])
        emb_subtype = ste(data[..., 1])
        emb_lvl = le(data[..., 2])
        emb_rspgrp = rspgrpe(data[..., 3])

        # concat activity category vectors together in last dimension
        dt_src = torch.cat([emb_type, emb_subtype, emb_lvl, emb_rspgrp], dim=2)
        dt_src = act_emb_layer_norm(dt_src)

        # mask any future sequences so attention will not use them
        mask = (torch.triu(torch.ones(bptt, bptt)) == 1).transpose(0, 1).to(device)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        ).to(device)

        # process static data
        tfmr_enc_out = tfmr_enc.forward(em_st_src)

        # forward pass main transfomer
        tfmr_out = tfmr_dec.forward(dt_src, memory=tfmr_enc_out, tgt_mask=mask)

        # get class probs of activity elements
        tclsprb = tc.forward(tfmr_out)
        stclsprb = stc.forward(tfmr_out)
        lclsprb = ltc.forward(tfmr_out)
        rspgrpsprb = rspgrpc.forward(tfmr_out)

        tclsprb = tclsprb.reshape(tclsprb.numel() // num_type_tokens, num_type_tokens)
        stclsprb = stclsprb.reshape(
            stclsprb.numel() // num_subtype_tokens, num_subtype_tokens
        )
        lclsprb = lclsprb.reshape(lclsprb.numel() // num_lvl_tokens, num_lvl_tokens)
        rspgrpsprb = rspgrpsprb.reshape(
            rspgrpsprb.numel() // num_rspgrp_tokens, num_rspgrp_tokens
        )

        tgt_loss = crit(tclsprb, tgt[..., 0].flatten())
        tgt_loss += crit(stclsprb, tgt[..., 1].flatten())
        tgt_loss += crit(lclsprb, tgt[..., 2].flatten())
        tgt_loss += crit(rspgrpsprb, tgt[..., 2].flatten())
        print(tgt_loss)
        tgt_loss.backward()
        # _=torch.nn.utils.clip_grad_norm_(
        #     list(
        #         itertools.chain.from_iterable(
        #             [
        #                 list(i)
        #                 for i in [
        #                     tc.parameters(),
        #                     stc.parameters(),
        #                     ltc.parameters(),
        #                     tfmr_dec.parameters(),
        #                     te.parameters(),
        #                     ste.parameters(),
        #                     le.parameters(),
        #                     act_emb_layer_norm.parameters(),
        #                     tfmr_enc.parameters(),
        #                     rspgrpe.parameters(),
        #                     rspgrpc.parameters(),
        #                     bert_squeeze_layer_norm.parameters(),
        #                     bertsqueeze.parameters(),
        #                     tmfr_out_layer_norm.parameters(),
        #                 ]
        #             ]
        #         )
        #     ),
        #     0.5,
        # )
        optimizer.step()
        db_optim.step()
        optimizer.zero_grad()
        total_loss += tgt_loss

        if counter % log_interval == 0:
            print(f"Epoch: {i}")
            print(f"LR: {scheduler.get_lr()[0]}")
            print(f"Loss: {(total_loss / log_interval)}")
            total_loss = 0.0

        if counter % log_interval * 5 == 0:
            print(
                tclsprb.argmax(dim=-1),
                stclsprb.argmax(dim=-1),
                lclsprb.argmax(dim=-1),
                rspgrpsprb.argmax(dim=-1),
            )
            print(tgt)

        counter += 1

    scheduler.step()

# turn on eval
tfmr_dec.eval().to(device)
# tfmr_enc.eval().to(device)
te.eval().to(device)
ste.eval().to(device)
le.eval().to(device)
ltc.eval().to(device)
stc.eval().to(device)
tc.eval().to(device)
static_model.eval().to(device)

# "validate" we have memorized data acceptably
with torch.no_grad():
    k = 0
    src = data[0]
    tgt = tgt[1]
    emb_type = te(src[..., 0])
    emb_subtype = ste(src[..., 1])
    emb_lvl = le(src[..., 2])
    emb_rspgrp = rspgrpe(src[..., 3])
    dt_src = torch.cat([emb_type, emb_subtype, emb_lvl, emb_rspgrp], dim=1)
    dt_src = torch.reshape(dt_src, (1, 2, 400))
    tfmr_out = tfmr_dec(dt_src, memory=em_st_src)

    tclsprb = tc.forward(tfmr_out)
    stclsprb = stc.forward(tfmr_out)
    lclsprb = ltc.forward(tfmr_out)
    rspgrpsprb = rspgrpc.forward(tfmr_out)

    tclsprb = tclsprb.reshape(2, num_type_tokens)
    stclsprb = stclsprb.reshape(2, num_subtype_tokens)
    lclsprb = lclsprb.reshape(2, num_lvl_tokens)
    rspgrpsprb = rspgrpsprb.reshape(2, num_rspgrp_tokens)

    tgt_loss = crit(tclsprb, tgt[..., 0].flatten())
    tgt_loss += crit(stclsprb, tgt[..., 1].flatten())
    tgt_loss += crit(lclsprb, tgt[..., 2].flatten())

    print(tgt_loss)
    print(tclsprb.argmax(dim=-1), stclsprb.argmax(dim=-1), lclsprb.argmax(dim=-1))
    print(tgt)
