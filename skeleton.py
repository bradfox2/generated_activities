import torch
import torch.nn as nn
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data = [
    [
        [["sos", "sos", "sos"], ["tx0", "stx0", "lx0"]], # lag      #--|
                                                                    #  | - mini batch
        [["sos1", "sos1", "sos1"], ["ty0", "sty0", "ly0"]],         #--| 
    ],
    [                                                    # ex seperate cr
        [["tx0", "stx0", "lx0"], ["tx1", "stx1", "lx1"]],
        [["ty0", "sty0", "ly0"], ["ty1", "sty1", "ly1"]],
    ],
    [
        [["tx1", "stx1", "lx1"], ["tx2", "stx2", "lx2"]],
        [["ty1", "sty1", "ly1"], ["ty2", "sty2", "ly2"]],
    ],
    [
        [["tx2", "stx2", "lx2"], ["eos", "eos", "eos"]],
        [["ty2", "sty2", "ly2"], ["eos1", "eos1", "eos1"]],
    ],
]


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


td = generate_token_dicts(data, 3)
t_td, st_td, l_td = td
n_tokens = len(t_td)

data_array = np.array(data)
for i in range(data_array.shape[-1]):
    token_dict = td[i]
    for field in np.nditer(data_array[..., i], op_flags=["readwrite"]):
        field[...] = int(token_dict[str(field)])

intv = np.vectorize(int)
data_array = intv(data_array)
# make 4d tensor of (cr x batch x activities lookback window(lag/bptt) x activity categories)


emb_dim = 100
num_seqences_into_tran = 300
num_attn_heads = 2
num_dec_layers = 2
# dims (mini_batch(batch_sz) x bptt x act_cats)
bptt = 2  # num lagged activities
batch_sz = 2  # num crs in mini batch

te = nn.Embedding(n_tokens, emb_dim).to(device)
e = nn.Embedding(n_tokens, emb_dim).to(device)
ste = nn.Embedding(n_tokens, emb_dim).to(device)
le = nn.Embedding(n_tokens, emb_dim).to(device)
tfmr_enc_l = nn.TransformerDecoderLayer(num_seqences_into_tran, num_attn_heads).to(
    device
)
tfmr_enc = nn.TransformerDecoder(tfmr_enc_l, num_dec_layers).to(device)
tc = nn.Linear(num_seqences_into_tran, len(t_td)).to(device)
stc = nn.Linear(num_seqences_into_tran, len(st_td)).to(device)
ltc = nn.Linear(num_seqences_into_tran, len(l_td)).to(device)
crit = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(
    [
        {"params": tc.parameters()},
        {"params": stc.parameters()},
        {"params": ltc.parameters()},
        {"params": tfmr_enc.parameters()},
        {"params": te.parameters()},
        {"params": ste.parameters()},
        {"params": le.parameters()},
        {"params": e.parameters()},
    ]
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# tfmr.eval()
tfmr_enc.train().to(device)

data_ten = torch.tensor(data_array).long().to(device)

i = 0
epochs = 50
for i in range(epochs):

    print(i)
    tgt_loss = 0.0
    optimizer.zero_grad()

    # k = 0
    for k in range(len(data_ten) - 1):
        src = data_ten[k, :, :, :] #
        tgt = data_ten[k + 1, :, :, :]

        embedded_src = e(src)

        # concat activity category vectors together in last dimension
        dt_src = torch.reshape(embedded_src, (bptt, batch_sz, src.shape[-1] * emb_dim))
        mask = (torch.triu(torch.ones(2, 2)) == 1).transpose(0, 1).to(device)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        ).to(device)
        tfmr_out = tfmr_enc.forward(
            dt_src, memory=torch.zeros(dt_src.shape).to(device), tgt_mask=mask
        )

        tclsprb = tc.forward(tfmr_out)
        stclsprb = stc.forward(tfmr_out)
        lclsprb = ltc.forward(tfmr_out)

        tclsprb = tclsprb.reshape(4, n_tokens)
        stclsprb = stclsprb.reshape(4, n_tokens)
        lclsprb = lclsprb.reshape(4, n_tokens)

        tgt_loss += crit(tclsprb, tgt[..., 0].flatten())
        tgt_loss += crit(stclsprb, tgt[..., 1].flatten())
        tgt_loss += crit(lclsprb, tgt[..., 2].flatten())

        tgt_loss.backward()
        optimizer.step()
        print(tgt_loss)

        tgt_loss = 0.0

    if i % 10 == 0:
        print(tclsprb.argmax(dim=-1), stclsprb.argmax(dim=-1), lclsprb.argmax(dim=-1))

tfmr_enc.eval().to(device)
te.eval().to(device)
ste.eval().to(device)
le.eval().to(device)
e.eval().to(device)
ltc.eval().to(device)
stc.eval().to(device)
tc.eval().to(device)

with torch.no_grad():
    k = 1
    src = data_ten[0, :, :, :]
    tgt = data_ten[1, :, :, :]
    embedded_src = e(src)
    dt_src = torch.reshape(embedded_src, (2, 2, 300))
    tfmr_out = tfmr_enc(dt_src, memory=torch.zeros(dt_src.shape).to(device))

    tclsprb = tc.forward(tfmr_out)
    stclsprb = stc.forward(tfmr_out)
    lclsprb = ltc.forward(tfmr_out)

    tclsprb = tclsprb.reshape(4, n_tokens)
    stclsprb = stclsprb.reshape(4, n_tokens)
    lclsprb = lclsprb.reshape(4, n_tokens)

    tgt_loss = crit(tclsprb, tgt[..., 0].flatten())
    tgt_loss += crit(stclsprb, tgt[..., 1].flatten())
    tgt_loss += crit(lclsprb, tgt[..., 2].flatten())

    print(tgt_loss)
    print(tclsprb.argmax(dim=-1), stclsprb.argmax(dim=-1), lclsprb.argmax(dim=-1))
    print(tgt)

