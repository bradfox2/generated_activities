import datetime
import logging
import pickle

import numpy as np
import torch
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
from utils import field_accuracy, set_seed, field_printer
import sys

set_seed(0)

load_chkpnt = True

model_name = "SIAG4"  # Seq_Ind_Acts_Generation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# create file handler which logs even debug messages
fh = logging.FileHandler(f"{model_name}_training.log")
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
# ch.setFormatter(formatter)
# add the handlers to the train_logger
logger.addHandler(fh)
# logger.addHandler(ch)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trnseq, tstseq, trnstat, tststat = get_dat_data(split_frac=0.8)

sequence_length = (
    5  # maximum number of independent category groups that make up a sequence
)
num_act_cats = 4  # number of independent fields in a category group
batch_sz = 12  # minibatch size, sequences of independent cat groups to be processed in parallel
rec_len = len(trnseq) // batch_sz  # num records in training set, used for batchifying
emb_dim = 192  # embedding dim for each categorical
embedding_dim_into_tran = (
    emb_dim * num_act_cats
)  # embedding dim size into transformer layers
tfmr_num_hidden = emb_dim * 4  # number of hidden units in transfomer linear layer
num_attn_heads = 8  # number of transformer attention heads
num_dec_layers = 4  # number of transformer decoder layers (main layers)
bptt = (
    sequence_length + 2
)  # back prop through time or sequence length, how long the sequence is that we are working with
num_epochs = 250

# tokenize, truncate, pad
(
    numer_trn_act_seqs,
    numer_tst_act_seqs,
    numer_trn_static_data,
    numer_tst_static_data,
) = process(trnseq, trnstat, tstseq, tststat, sequence_length + 1)

# check to ensure data is good
assert (
    len(trnseq[~trnseq.index.isin(numer_trn_act_seqs.index)]) == 0
), "Records are being dropped during processing"
assert trnseq.index.identical(
    numer_trn_act_seqs.index
), "Mismatch between seq tokenized data and seq numericalized data."
assert numer_trn_static_data.index.identical(
    numer_trn_act_seqs.index
), "Mismatch between static data and seq data."

seq_data_trn = batchify_act_seqs(numer_trn_act_seqs, batch_sz).contiguous().to(device)
seq_data_tst = batchify_act_seqs(numer_tst_act_seqs, batch_sz).contiguous().to(device)

static_data_trn = batchify_static_data(
    numer_trn_static_data[: seq_data_trn.shape[0] * batch_sz], batch_sz
)
static_data_tst = batchify_static_data(
    numer_tst_static_data[: seq_data_tst.shape[0] * batch_sz], batch_sz
)

# dims (mini_batch(batch_sz) x bptt x act_cats)
def gen_inp_data_set(seq_data: torch.Tensor, static_data: np.array):
    """generator that advances through the 'group-of-sequences' dimension, 
    one group at a time, generating sequence input and target sets and static data"""
    for i in range(len(seq_data)):
        inp = seq_data[i, 0:-1]
        target = seq_data[i, 1:]
        yield inp, target, static_data[i]


from typing import Tuple


def validate(eval_model, seq_data, static_data) -> Tuple[float, float]:
    eval_model.eval()
    with torch.no_grad():
        val_loss = 0.0
        data_gen = gen_inp_data_set(seq_data, static_data)
        # data, tgt, static_data = next(data_gen)
        val_acc_l = []
        for data, tgt, static_data in data_gen:
            static_data = static_tokenizer(
                static_data.tolist(), padding=True, truncation=True, return_tensors="pt"
            ).to(eval_model.device)
            preds = eval_model(data.to(eval_model.device), static_data)
            batch_loss = eval_model.loss(preds, tgt.to(eval_model.device))
            val_loss += batch_loss
            val_acc = [
                field_accuracy(field, preds[idx], tgt[..., idx], 3)
                for idx, field in enumerate(fields)
            ]
            val_acc_l.append(val_acc)
            logger.debug("Val Acc: {.5f}".format(val_acc))
        avg_val_acc = [sum(i) / len(i) for i in zip(*val_acc_l)]
        logger.info(f"Mean Val Acc: {avg_val_acc}")
        logger.info(
            [
                field_printer(field, preds[idx], tgt[..., idx])
                for idx, field in enumerate(fields)
            ]
        )
        return (val_loss.item() / len(seq_data), sum(avg_val_acc) / len(avg_val_acc))


fields = [TYPE, SUBTYPE, LVL, RESPGROUP]
total_trn_samples = len(trnseq)
type_ = IndependentCategorical.from_torchtext_field("type_", TYPE, total_trn_samples)
subtype = IndependentCategorical.from_torchtext_field(
    "subtype", SUBTYPE, total_trn_samples
)
lvl = IndependentCategorical.from_torchtext_field("lvl", LVL, total_trn_samples)
respgroup = IndependentCategorical.from_torchtext_field(
    "respgroup", RESPGROUP, total_trn_samples
)

model = SAModel(
    sequence_length=sequence_length,
    categorical_embedding_dim=emb_dim,
    num_attn_heads=num_attn_heads,
    num_hidden=tfmr_num_hidden,
    num_transformer_layers=num_dec_layers,
    learning_rate=1e-5,
    independent_categoricals=[type_, subtype, lvl, respgroup],
    freeze_static_model_weights=False,
    warmup_steps=(rec_len // batch_sz) * 1.5,  # about 1 epoch
    total_steps=num_epochs * (rec_len // batch_sz),
    device=device,
)

static_tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased",
    return_tensors="pt",
    vocab_file="./distilbert_weights/vocab.txt",
)


if load_chkpnt:  # continue training
    try:
        model_path = "./saved_models/chkpnt-SIAG3.ptm"
        logger.info(f"Loading model from {model_path}")
        model.to(device)  # move model before loading optimizer
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
    except:
        Warning("Can not load checkpoint, training from scratch.")
        epoch = 0
else:
    epoch = 0

pickle.dump(
    [TYPE, SUBTYPE, LVL, RESPGROUP],
    open(f"./saved_models/{model_name}_fields.pkl", "wb"),
)

model.to(device)
log_interval = 10
train_loss_record = []
val_acc_record = []
for i in range(epoch, num_epochs):
    model.train()
    epoch_loss = 0.0
    counter = 0
    loss_tracker = []
    data_gen = gen_inp_data_set(seq_data_trn, static_data_trn)
    for data, tgt, static_data_txt in data_gen:
        static_data = static_tokenizer(
            static_data_txt.tolist(), padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        batch_loss = model.learn(data.to(device), static_data, tgt.to(device))
        epoch_loss += batch_loss
        counter += 1
        if counter % log_interval == 0:
            loss_tracker.append(epoch_loss / log_interval)
            logger.info(f"Epoch: {i}")
            logger.info(f"Record: {counter}/{rec_len}")
            logger.info(f"LR: {model.scheduler.get_last_lr()[0]}")
            logger.info(f"Loss: {(epoch_loss / log_interval):.3f}")
            epoch_loss = 0.0
    epoch_avg_loss = sum(loss_tracker) / len(loss_tracker)

    train_loss_record.append(epoch_avg_loss)

    val_loss, val_acc = validate(model, seq_data_tst, static_data_tst)
    val_acc_record.append(val_acc)

    logger.info(f"Validation Loss: {val_loss:.3f}")
    logger.info(f"Valdation Accuracy: {val_acc:.3f}")

    # save checkpoint
    if val_acc > max(val_acc_record):
        checkpoint_path = f"./saved_models/chkpnt-{model_name}-EP{i}-TRNLOSS{str(epoch_avg_loss)[:5].replace('.','dot')}-{datetime.datetime.today().strftime('%Y-%m-%d %H-%M-%S')}.ptm"
        checkpoint_path = checkpoint_path[:260].replace(" ", "_")
        logger.info(f"Saving Checkpoint {checkpoint_path}")
        torch.save(
            {
                "epoch": i,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": model.optimizer.state_dict(),
                "scheduler_state_dict": model.scheduler.state_dict(),
            },
            checkpoint_path,
        )


torch.save(model.state_dict(), f"./saved_models/{model_name}.ptm")


def load_model(device: torch.device):
    model = SAModel(
        sequence_length=sequence_length,
        categorical_embedding_dim=emb_dim,
        num_attn_heads=num_attn_heads,
        num_hidden=tfmr_num_hidden,
        num_transformer_layers=num_dec_layers,
        learning_rate=1e-3,
        independent_categoricals=[type_, subtype, lvl, respgroup],
        freeze_static_model_weights=True,
        device=device,
    )

    chkpnt = torch.load(
        "./saved_models/chkpnt-SIAG3-EP40-TRNLOSS7dot716-2020-08-21_23-46-15.ptm",
        map_location=device,
    )

    model.load_state_dict(chkpnt["model_state_dict"])
    model.eval()
    model.to(device)
    return model


# model = load_model(device)

# trnseq, tstseq, trnstat, tststat = get_dat_data(split_frac=0.8)
# (
#     numer_trn_act_seqs,
#     numer_tst_act_seqs,
#     numer_trn_static_data,
#     numer_tst_static_data,
# ) = process(trnseq, trnstat, tstseq, tststat, 5 + 1)

# trnseq.apply(truncate_series_by_len, args=(6,))

# trnseq[0][:6]
