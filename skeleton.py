""" toy example of next sequence prediction at the intra-sequential element level using 
transfomer with sequence level static data

ie. ['sos','sos'] -> ['t1', 's1'] where t and s are from different categorical sets, 
but where value of  s1 is dependent on t1
"""

import torch
import torch.nn as nn
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data is 1 batches of bptt 5, minibatch size 2,  4 act categories
data = [
    [
        ["sos", "sos", "sos", "sos"],
        ["tx0", "stx0", "lx0", "rgx0"],
        ["tx1", "stx1", "lx1", "rgx1"],
        ["tx2", "stx2", "lx2", "rgx2"],
        ["eos", "eos", "eos", "eos"],
    ],  #           --|
    #                                                               | - mini batch
    [
        ["sos", "sos", "sos", "sos"],
        ["ty0", "sty0", "ly0", "rgy0"],
        ["ty1", "sty1", "ly1", "rgy1"],
        ["ty2", "sty2", "ly2", "rgy2"],
        ["eos", "eos", "eos", "rgy3"],
    ],  #           --|
]

# toy static data, unique token sequences
# this should help model seperate the sos -> tx0/ty0 sequence prediction
static_data = [[1, 2, 3, 4], [6, 7, 8, 9]]


def generate_token_dicts(data, ind_tok_dim):
    """generate independent token dicts for data along ind_tok_dim"""
    tokenizers = []
    data_array = np.array(data)
    for act_cat_field in data_array.swapaxes(ind_tok_dim, 0).swapaxes(1, ind_tok_dim):
        tokenizer = {}
        for i in np.nditer(act_cat_field.flat):
            token = str(i.item())
            if token not in tokenizer.keys():
                tokenizer[token] = int(len(tokenizer))
        tokenizers.append(tokenizer)
    return tokenizers


# generate independent token dicts for categorical field labels
td = generate_token_dicts(data, 2)
t_td, st_td, l_td, rg_td = td
n_tokens = len(t_td)
n_act_cats = len(data[0][0])

# numericalize arrays based on token dicts
tok_data_array = np.array(data)
for i in range(tok_data_array.shape[-1]):
    token_dict = td[i]
    for field in np.nditer(tok_data_array[..., i], op_flags=["readwrite"]):
        field[...] = str(field)

# arrange array, but with text tokens for reference use
data_array = np.array(data)
for i in range(data_array.shape[-1]):
    token_dict = td[i]
    for field in np.nditer(data_array[..., i], op_flags=["readwrite"]):
        field[...] = int(token_dict[str(field)])

intv = np.vectorize(int)
data_array = intv(data_array)

emb_dim = 100
num_attn_heads = 2
num_dec_layers = 2
# dims (mini_batch(batch_sz) x bptt x act_cats)
bptt = 4  # num lagged activities
mb_sz = 2
tran_dim = emb_dim * n_act_cats

e = nn.Embedding(n_tokens, emb_dim).to(device)
## TODO: validate seperate embedding per act category works better
te = nn.Embedding(n_tokens, emb_dim).to(device)
ste = nn.Embedding(n_tokens, emb_dim).to(device)
le = nn.Embedding(n_tokens, emb_dim).to(device)
##
se = nn.Embedding(11, 400).to(device)
tfmr_dec_l = nn.TransformerDecoderLayer(tran_dim, num_attn_heads).to(device)
tfmr_dec = nn.TransformerDecoder(tfmr_dec_l, num_dec_layers).to(device)
tfmr_enc_l = nn.TransformerEncoderLayer(tran_dim, 1).to(device)
tfmr_enc = nn.TransformerEncoder(tfmr_enc_l, 1).to(device)
tc = nn.Linear(tran_dim, len(t_td)).to(device)
stc = nn.Linear(tran_dim, len(st_td)).to(device)
ltc = nn.Linear(tran_dim, len(l_td)).to(device)
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
        {"params": e.parameters()},
        {"params": tfmr_enc.parameters()},
    ]
)

tfmr_dec.train().to(device)

# make 4d tensor of (bpttbatch x minibatch x activities lookback window(seq lengths) x activity categories)
data_ten = torch.tensor(data_array).long().to(device)

# simple static data tensors
st_data_ten = torch.tensor(static_data).long().to(device)

i = 0
epochs = 500  # 30 seems enough to memorize this toy set
# TODO: static data does not seem to make a difference in model differentiating tx0/ty0, and about 40 epochs loss plataeus
for i in range(epochs):

    print(i)
    tgt_loss = 0.0
    optimizer.zero_grad()

    # k = 0
    for k in range(len(data_ten) - 1):
        # iterate batches of bptt sequences, minibatches processed in parallel on dim 1
        src = data_ten[:, k : k + bptt, :]
        tgt = data_ten[:, k + 1 : k + 1 + bptt, :]

        em_st_src = se(st_data_ten.t()).to(device)
        embedded_src = e(src)

        # concat activity category vectors together in last dimension
        dt_src = torch.reshape(embedded_src, (mb_sz, bptt, src.shape[-1] * emb_dim))

        # mask any future sequences so attention will not use them
        mask = (torch.triu(torch.ones(mb_sz, mb_sz)) == 1).transpose(0, 1).to(device)
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

        tclsprb = tclsprb.reshape(bptt * mb_sz, n_tokens)
        stclsprb = stclsprb.reshape(bptt * mb_sz, n_tokens)
        lclsprb = lclsprb.reshape(bptt * mb_sz, n_tokens)

        tgt_loss += crit(tclsprb, tgt[..., 0].flatten())
        tgt_loss += crit(stclsprb, tgt[..., 1].flatten())
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
    src = data_ten[:, 0, :]
    tgt = data_ten[:, 1, :]
    embedded_src = e(src)
    dt_src = torch.reshape(embedded_src, (mb_sz, 1, src.shape[-1] * emb_dim))

    em_st_src = se(st_data_ten.t()).to(device)
    tfmr_enc_out = tfmr_enc.forward(em_st_src)

    tfmr_out = tfmr_dec(dt_src, memory=tfmr_enc_out)

    tclsprb = tc.forward(tfmr_out)
    stclsprb = stc.forward(tfmr_out)
    lclsprb = ltc.forward(tfmr_out)

    tclsprb = tclsprb.reshape(mb_sz, n_tokens)
    stclsprb = stclsprb.reshape(mb_sz, n_tokens)
    lclsprb = lclsprb.reshape(mb_sz, n_tokens)

    if tclsprb.shape[1] == 1:
        tgt_loss = crit(tclsprb.squeeze(), tgt[..., 0].flatten())
        tgt_loss += crit(stclsprb.squeeze(), tgt[..., 1].flatten())
        tgt_loss += crit(lclsprb.squeeze(), tgt[..., 2].flatten())

    print(tgt_loss)
    print(tclsprb.argmax(dim=-1), stclsprb.argmax(dim=-1), lclsprb.argmax(dim=-1))
    print(tgt)

# infer
with torch.no_grad():
    mb_sz = 1
    k = 0
    src = data_ten[0, 0, :].unsqueeze(0)
    tgt = data_ten[0, 1, :]
    embedded_src = e(src)
    dt_src = torch.reshape(embedded_src, (mb_sz, 1, src.shape[-1] * emb_dim))

    em_st_src = se(st_data_ten[0, :].t()).unsqueeze(0).to(device)
    tfmr_enc_out = tfmr_enc.forward(em_st_src)

    tfmr_out = tfmr_dec(dt_src, memory=tfmr_enc_out)

    tclsprb = tc.forward(tfmr_out)
    stclsprb = stc.forward(tfmr_out)
    lclsprb = ltc.forward(tfmr_out)

    tclsprb = tclsprb.reshape(mb_sz, n_tokens)
    stclsprb = stclsprb.reshape(mb_sz, n_tokens)
    lclsprb = lclsprb.reshape(mb_sz, n_tokens)
