import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data = [
    [
        [["sos", "sos", "sos"], ["tx0", "stx0", "lx0"]],
        [["sos", "sos", "sos"], ["ty0", "sty0", "ly0"]],
    ],
    [
        [["tx0", "stx0", "lx0"], ["tx1", "stx1", "lx1"]],
        [["ty0", "sty0", "ly0"], ["ty1", "sty1", "ly1"]],
    ],
    [
        [["tx1", "stx1", "lx1"], ["tx2", "stx2", "lx2"]],
        [["ty1", "sty1", "ly1"], ["ty2", "sty2", "ly2"]],
    ],
    [
        [["tx2", "stx2", "lx2"], ["eos", "eos", "eos"]],
        [["ty2", "sty2", "ly2"], ["eos", "eos", "eos"]],
    ],
]


def generate_token_dicts(data, ind_tok_dim):
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

data_array = np.array(data)
for i in range(data_array.shape[-1]):
    token_dict = td[i]
    for field in np.nditer(data_array[..., i], op_flags=["readwrite"]):
        field[...] = int(token_dict[str(field)])

intv = np.vectorize(int)
data_array = intv(data_array)

# make 5d tensor of (sequential info x batch x activities lookback window x activity categories x activity category vector)
te = nn.Embedding(8, 100).to(device)
e = nn.Embedding(8, 100).to(device)
ste = nn.Embedding(8, 100).to(device)
le = nn.Embedding(8, 100).to(device)
tfmr = nn.Transformer(300, 2, 2, 2).to(device)
tc = nn.Linear(300, len(t_td)).to(device)
stc = nn.Linear(300, len(st_td)).to(device)
ltc = nn.Linear(300, len(l_td)).to(device)
crit = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(
    [
        {"params": tc.parameters()},
        {"params": stc.parameters()},
        {"params": ltc.parameters()},
        {"params": tfmr.parameters()},
        {"params": te.parameters()},
        {"params": ste.parameters()},
        {"params": le.parameters()},
        {"params": e.parameters()},
    ]
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# tfmr.eval()
tfmr.train().to(device)

data_ten = torch.tensor(data_array).long().to(device)

for i in range(50):

    print(i)
    tgt_loss = 0.0
    optimizer.zero_grad()

    # k = 0
    for k in range(3):
        src = data_ten[k, :, :, :]
        tgt = data_ten[k + 1, :, :, :]

        embedded_src = e(src)
        embedded_tgt = e(tgt)

        # torch.cat((te,ste,le))

        # concat activity category vectors together in last dimension
        dt_src = torch.reshape(embedded_src, (2, 2, 300))
        dt_tgt = torch.reshape(embedded_tgt, (2, 2, 300))
        mask = (torch.triu(torch.ones(2, 2)) == 1).transpose(0, 1).to(device)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        tfmr_out = tfmr.forward(dt_src, dt_tgt, src_mask=mask)

        tclsprb = tc.forward(tfmr_out)
        stclsprb = stc.forward(tfmr_out)
        lclsprb = ltc.forward(tfmr_out)

        tclsprb = tclsprb.reshape(4, 8)
        stclsprb = stclsprb.reshape(4, 8)
        lclsprb = lclsprb.reshape(4, 8)

        tgt_loss = crit(tclsprb, tgt[..., 0].flatten())
        tgt_loss += crit(stclsprb, tgt[..., 1].flatten())
        tgt_loss += crit(lclsprb, tgt[..., 2].flatten())

        tgt_loss.backward()
        optimizer.step()

    print(tgt_loss)

    print(tclsprb.argmax(dim=-1), stclsprb.argmax(dim=-1), lclsprb.argmax(dim=-1))

tfmr.eval()
k = 1
src = data_ten[0, :, :, :]
tgt = data_ten[1, :, :, :]
embedded_src = e(src)
dt_src = torch.reshape(embedded_src, (2, 2, 300))
tfmr_out = tfmr.decoder(tgt=dt_src, memory=torch.zeros(2, 2, 300).to(device))

tclsprb = tc.forward(tfmr_out)
stclsprb = stc.forward(tfmr_out)
lclsprb = ltc.forward(tfmr_out)

tclsprb = tclsprb.reshape(4, 8)
stclsprb = stclsprb.reshape(4, 8)
lclsprb = lclsprb.reshape(4, 8)

tgt_loss = crit(tclsprb, tgt[..., 0].flatten())
tgt_loss += crit(stclsprb, tgt[..., 1].flatten())
tgt_loss += crit(lclsprb, tgt[..., 2].flatten())

print(tgt_loss)


print(tclsprb.argmax(dim=-1), stclsprb.argmax(dim=-1), lclsprb.argmax(dim=-1))
