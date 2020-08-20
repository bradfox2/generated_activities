import datetime
import logging
import pickle
from utils import set_seed

set_seed()
import numpy as np


import torch


from torch.nn.modules import loss
from transformers import DistilBertTokenizer

from data_processing import (
    LVL,
    RESPGROUP,
    SUBTYPE,
    TYPE,
    batchify_act_seqs,
    batchify_static_data,
    process,
)
from load_staged_acts import get_dat_data
from model import IndependentCategorical, SAModel
from utils import field_printer

torch.manual_seed(0)

model_name = "SIAG_very_small"  # Seq_Ind_Acts_Generation

train_logger = logging.getLogger(model_name)
train_logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(f"{model_name}_training.log")
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the train_logger
train_logger.addHandler(fh)
train_logger.addHandler(ch)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trnseq, tstseq, trnstat, tststat = get_dat_data(split_frac=0.8)

sequence_length = (
    5  # maximum number of independent category groups that make up a sequence
)
num_act_cats = 4  # number of independent fields in a category group
batch_sz = 32  # minibatch size, sequences of independent cat groups to be processed in parallel
rec_len = len(trnseq) // batch_sz  # num records in training set, used for batchifying
emb_dim = 16  # embedding dim for each categorical
embedding_dim_into_tran = (
    emb_dim * num_act_cats
)  # embedding dim size into transformer layers
num_attn_heads = 1  # number of transformer attention heads
num_dec_layers = 1  # number of transformer decoder layers (main layers)
bptt = sequence_length  # back prop through time or sequence length, how far the lookback window goes

# tokenize, truncate, pad
(
    numer_trn_act_seqs,
    numer_tst_act_seqs,
    numer_trn_static_data,
    numer_tst_static_data,
) = process(trnseq, trnstat, tstseq, tststat, sequence_length + 1)

assert len(numer_trn_act_seqs) == len(numer_trn_static_data)
assert numer_trn_act_seqs.index[0] == numer_trn_static_data.index[0]

seq_data_trn = batchify_act_seqs(numer_trn_act_seqs, batch_sz).contiguous().to(device)
seq_data_tst = batchify_act_seqs(numer_tst_act_seqs, batch_sz).contiguous().to(device)

static_data_trn = batchify_static_data(
    numer_trn_static_data[: seq_data_trn.shape[0] * batch_sz], batch_sz
)
static_data_tst = batchify_static_data(
    numer_trn_static_data[: seq_data_trn.shape[0] * batch_sz], batch_sz
)

# dims (mini_batch(batch_sz) x bptt x act_cats)
def gen_inp_data_set(seq_data: torch.Tensor, static_data: np.array):
    """generator that advances through the 'group-of-sequences' dimension, 
    one group at a time, generating sequence input and target sets and static data"""
    for i in range(len(seq_data)):
        inp = seq_data[i, 0:-1]
        target = seq_data[i, 1:]
        yield inp, target, static_data[i]


def validate():
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        data_gen = gen_inp_data_set(seq_data_tst, static_data_tst)
        # data, tgt, static_data = next(data_gen)
        for data, tgt, static_data in data_gen:
            static_data = static_tokenizer(
                static_data.tolist(), padding=True, truncation=True, return_tensors="pt"
            ).to(model.device)
            preds = model(data.to(model.device), static_data)
            batch_loss = model.loss(preds, tgt.to(model.device))
            val_loss += batch_loss
            # field_printer(TYPE, preds[0], tgt[..., 0])
            # field_printer(SUBTYPE, preds[1], tgt[..., 1])
            # field_printer(LVL, preds[2], tgt[..., 2])
            # field_printer(RESPGROUP, preds[3], tgt[..., 3])
        return val_loss.item() / len(seq_data_tst)


type_ = IndependentCategorical.from_torchtext_field("type_", TYPE)
subtype = IndependentCategorical.from_torchtext_field("subtype", SUBTYPE)
lvl = IndependentCategorical.from_torchtext_field("lvl", LVL)
respgroup = IndependentCategorical.from_torchtext_field("respgroup", RESPGROUP)


model = SAModel(
    sequence_length,
    emb_dim,
    num_attn_heads,
    num_dec_layers,
    learning_rate=1e-3,
    learning_rate_decay_rate=0.98,
    independent_categoricals=[type_, subtype, lvl, respgroup],
    freeze_static_model_weights=True,
    p_class_drop=0.1,
    device=device,
)

static_tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased", return_tensors="pt"
)


model.to(device)
log_interval = 100
train_loss_record = []
epochs = 10
for i in range(epochs):
    model.train()
    epoch_loss = 0.0
    counter = 0
    loss_tracker = []
    data_gen = gen_inp_data_set(seq_data_trn, static_data_trn)
    for data, tgt, static_data in data_gen:
        static_data = static_tokenizer(
            static_data.tolist(), padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        batch_loss = model.learn(data, static_data, tgt)
        epoch_loss += batch_loss
        counter += 1
        if counter % log_interval == 0:
            loss_tracker.append(epoch_loss / log_interval)
            train_logger.info(f"Epoch: {i}")
            train_logger.info(f"Record: {counter}/{rec_len}")
            train_logger.info(f"LR: {model.scheduler.get_last_lr()[0]}")
            train_logger.info(f"Loss: {(epoch_loss / log_interval):.3f}")
            epoch_loss = 0.0
    epoch_avg_loss = sum(loss_tracker) / len(loss_tracker)
    train_loss_record.append(epoch_avg_loss)

    train_logger.info(f"Validation Loss: {validate():.3f}")

    # save checkpoint
    checkpoint_path = f"./saved_models/chkpnt-{model_name}-EP{i}-TRNLOSS{str(epoch_avg_loss)[:5].replace('.','dot')}-{datetime.datetime.today().strftime('%Y-%m-%d %H-%M-%S')}.ptm"
    checkpoint_path = checkpoint_path[:260].replace(" ", "_")
    train_logger.info(f"Saving Checkpoint {checkpoint_path}")
    torch.save(
        {
            "epoch": i,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": model.optimizer.state_dict(),
            "static_optimizer_state_dict": model.static_optimizer.state_dict(),
            "loss": epoch_avg_loss,
        },
        checkpoint_path,
    )

torch.save(model.state_dict(), f"./saved_models/{model_name}.ptm")
pickle.dump(
    [TYPE, SUBTYPE, LVL, RESPGROUP],
    open(f"./saved_models/{model_name}_fields.pkl", "wb"),
)


def load_model(device: torch.device):
    model = SAModel(
        sequence_length,
        batch_sz,
        emb_dim,
        num_attn_heads,
        num_dec_layers,
        learning_rate=1e-3,
        learning_rate_decay_rate=0.98,
        independent_categoricals=[type_, subtype, lvl, respgroup],
        device=device,
    )

    chkpnt = torch.load(
        "./saved_models/chkpnt-SIAG-EP2-TRNLOSS8dot439-2020-08-19_17-43-19.ptm",
        map_location=device,
    )

    model.load_state_dict(chkpnt["model_state_dict"])
    model.eval()
    model.to(device)
    return model
