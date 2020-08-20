import pickle

import pandas
import torch
from transformers import DistilBertTokenizer
from model import SAModel, IndependentCategorical

from data_processing import (
    LVL,
    RESPGROUP,
    SUBTYPE,
    TYPE,
    numericalize,
)

# model parameters from trainig script or as args
sequence_length = (
    5  # maximum number of independent category groups that make up a sequence
)
num_act_cats = 4  # number of independent fields in a category group
batch_sz = (
    8  # minibatch size, sequences of independent cat groups to be processed in parallel
)
emb_dim = 128  # embedding dim for each categorical
embedding_dim_into_tran = (
    emb_dim * num_act_cats
)  # embedding dim size into transformer layers
num_attn_heads = 4  # number of transformer attention heads
num_dec_layers = 2  # number of transformer decoder layers (main layers)
bptt = sequence_length  # back prop through time or sequence length, how far the lookback window goes


model_name = "SIAG_small"
fields = f"{model_name}_fields.pkl"

# load model and fields(tokenizers) needed
model = torch.load(f"./saved_models/{model_name}.ptm", map_location="cpu")

TYPE, SUBTYPE, LVL, RESPGROUP = pickle.load(open(f"./saved_models/{fields}", "rb"))

static_tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased", return_tensors="pt"
)

type_ = IndependentCategorical.from_torchtext_field("type_", TYPE)
subtype = IndependentCategorical.from_torchtext_field("subtype", SUBTYPE)
lvl = IndependentCategorical.from_torchtext_field("lvl", LVL)
respgroup = IndependentCategorical.from_torchtext_field("respgroup", RESPGROUP)

model = SAModel(
    sequence_length,
    batch_sz,
    emb_dim,
    num_attn_heads,
    num_dec_layers,
    learning_rate=1e-3,
    learning_rate_decay_rate=0.98,
    independent_categoricals=[type_, subtype, lvl, respgroup],
    device=torch.device("cpu"),
)

model_state_dict = torch.load("./saved_models/SIAG_small.ptm", map_location="cpu",)
model.load_state_dict(model_state_dict)

static_data = "the pump broke really bad in the containment building"

static_data = static_tokenizer(
    [static_data], padding=True, truncation=True, return_tensors="pt"
)

seq_data = pandas.Series([[["<sos>"] * 4]]).apply(
    numericalize, args=[TYPE, SUBTYPE, LVL, RESPGROUP]
)
seq_data = torch.tensor(seq_data)

model.eval()
with torch.no_grad():
    t, st, l, rg = None, None, None, None
    while t is not TYPE.vocab.stoi["<eos>"] and len(seq_data) <= 6:
        t, st, l, rg = model(seq_data, static_data)
        t = t[-1].argmax()
        st = st[-1].argmax()
        l = l[-1].argmax()
        rg = rg[-1].argmax()
        next_seq = torch.tensor([[[t, st, l, rg]]])
        seq_data = torch.cat((seq_data, next_seq))
    print(seq_data)

