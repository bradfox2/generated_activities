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
categorical_embedding_dim = 192  # embedding dim for each categorical
# embedding_dim_into_tran = (
#     emb_dim * num_act_cats
# )  # embedding dim size into transformer layers
num_attn_heads = 8  # number of transformer attention heads
num_transformer_layers = 4  # number of transformer decoder layers (main layers)
num_hidden = categorical_embedding_dim * 4
bptt = (
    sequence_length + 2
)  # back prop through time or sequence length, how far the lookback window goes


model_name = "SIAG4"
fields = f"{model_name}_fields.pkl"

# load model and fields(tokenizers) needed
model = torch.load(f"./saved_models/{model_name}.ptm", map_location="cpu")

TYPE, SUBTYPE, LVL, RESPGROUP = pickle.load(open(f"./saved_models/{fields}", "rb"))

static_tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased", return_tensors="pt"
)

type_ = IndependentCategorical.from_torchtext_field_wrapper("type_", TYPE, 1)
subtype = IndependentCategorical.from_torchtext_field_wrapper("subtype", SUBTYPE, 1)
lvl = IndependentCategorical.from_torchtext_field_wrapper("lvl", LVL, 1)
respgroup = IndependentCategorical.from_torchtext_field_wrapper(
    "respgroup", RESPGROUP, 1
)

model = SAModel(
    sequence_length,
    categorical_embedding_dim,
    num_attn_heads,
    num_hidden,
    num_transformer_layers,
    learning_rate=1e-3,
    independent_categoricals=[type_, subtype, lvl, respgroup],
    freeze_static_model_weights=True,
    warmup_steps=1,
    total_steps=1,
    static_data_embedding_size=768,
    dropout=0.2,
    grad_norm_clip=1.0,
    device=torch.device("cpu"),
)

model_state_dict = torch.load(
    f"./saved_models/{model_name}.ptm",
    map_location="cpu",
)
model.load_state_dict(model_state_dict)

from data.get_data import get_cr_feature_data


def predict(cr_cd, cap_class, resp_group):

    feature_data = get_cr_feature_data(cr_cd, cap_class, resp_group)[0]

    static_data = static_tokenizer(
        [feature_data], padding=True, truncation=True, return_tensors="pt"
    )

    seq_data = pandas.Series([[["<sos>"] * 4]]).apply(
        numericalize, args=[TYPE, SUBTYPE, LVL, RESPGROUP]
    )
    seq_data = torch.tensor(seq_data)

    model.eval()
    with torch.no_grad():
        t, st, l, rg = None, None, None, None
        while (t.item() if t else None) != TYPE.vocab.stoi["<eos>"] and len(
            seq_data
        ) <= 6:
            t, st, l, rg = model(seq_data, static_data)
            t = t[-1].argmax()
            st = st[-1].argmax()
            l = l[-1].argmax()
            rg = rg[-1].argmax()
            next_seq = torch.tensor([[[t, st, l, rg]]])
            seq_data = torch.cat((seq_data, next_seq))
            print(
                TYPE.vocab.itos[t],
                SUBTYPE.vocab.itos[st],
                LVL.vocab.itos[l],
                RESPGROUP.vocab.itos[rg],
            )
            print(feature_data)


predict("20-11517", "N", "8516")
