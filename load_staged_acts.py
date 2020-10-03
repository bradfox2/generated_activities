"""loads long dataset of staged acts with the cr details, and makes staged act sequences grouped by crs"""

from typing import Any, List

import numpy as np
import pandas as pd
import torch
import torchtext
from torch.utils.data import DataLoader, Dataset

from utils import set_seed

set_seed(0)


def create_act_seqs(df, seq_field_names, group_column_name="CR_CD"):
    return StagedActsDataset.create_act_seqs(df)


feature_cols = {
    "LI_QCLS_CD": str,
    "LI_FAIL_CD": str,
    "CAP_CLASS": str,
    "RESPONSIBLE_GROUP": str,
    "TITLE": str,
    "LI_SYS_ID": str,
    "LI_NAME": str,
    "PLNT_UNIT": str,
    "LI_EQUIP_COMP_CD": str,
    "LI_EQUIP": str,
    "IDENT_PER_ORG_UNIT": str,
    "DESCR": str,
    "LEADER_COMMENT": str,
    "ACTION_TAKEN_TEXT": str,
    "SUG_DISP": str,
}


class StagedActsDataset(Dataset):

    staged_activity_fields = {
        "SA_WRK_TYPE": object,
        "SA_SUB_TYPE": object,
        "SA_CAP_LVL": object,
        "SA_WRK_RESP_ORG_UNIT": object,
    }

    def __init__(self, csv_path, transform=None):
        self.raw_data = self.get_data(csv_path)
        self.data = self._process_raw_data()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Any:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data.iloc[idx]

        if self.transform:
            for f in self.transform:
                sample = f(sample)

        return sample

    def _process_raw_data(self):
        data = self.raw_data.copy(deep=True).drop_duplicates().reset_index()
        feats = data.groupby("CR_CD").first()
        return pd.concat([self.create_act_seqs(data), feats], axis=1)

    @classmethod
    def get_data(cls, csv_path) -> pd.DataFrame:
        return pd.read_csv(  # type: ignore
            csv_path,
            dtype=dict(cls.staged_activity_fields, **feature_cols),
            parse_dates=True,
            # keep most of the pandas default nan values, but not the N/A which
            # is what is used for acts that have 3/4 fields filled and one field
            # that is not applicable
            na_values=[
                "#N/A",
                "#N/A N/A",
                "#NA",
                "-1.#IND",
                "-1.#QNAN",
                "-NaN",
                "-nan",
                "1.#IND",
                "1.#QNAN",
                "<NA>",
                "NA",
                "NULL",
                "NaN",
                "nan",
                "null",
                None,
                "",
                " ",
                "  ",
                "?",
                "NONE",
            ],
            keep_default_na=False,
        )

    @classmethod
    def textify_feature_data(cls, cr_data, feature_cols):
        cr_data["TEXT"] = ""
        for col in feature_cols:
            cr_data["TEXT"] += f" [{col}] " + cr_data.get(col, "[SEP]").astype(str)
        return cr_data

    @classmethod
    def create_act_seqs(cls, df):
        """Create a list of list of column values specificed in seq_field_names, as grouped by group_column_name"""
        df["field_sequence"] = df[cls.staged_activity_fields].values.tolist()
        act_seqs = df.groupby("CR_CD")["field_sequence"].apply(list)
        return act_seqs


class Textify(object):
    """ create a text column in the data that concats feature cols with tags"""

    def __init__(self, feature_cols):
        self.feature_cols = feature_cols

    def __call__(self, data) -> pd.DataFrame:
        data["TEXT"] = ""
        for col in self.feature_cols:
            data["TEXT"] += f" [{col}] " + data.get(col, "[SEP]").fillna("[SEP]")
        return data


class TTFieldWrapper(torchtext.data.Field):
    """adds a min frequency property to the torchtext.data.Field class"""

    def __init__(self, min_frequency: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_frequency = min_frequency


class StagedActsDatasetProcessor:
    """trains torchtext fields on the sequences of ind cat categories.
    fields should be positioned in the passed list in the order of the sequence category"""

    def __init__(
        self,
        dataset: StagedActsDataset,
        fields: List[TTFieldWrapper],
        max_len: int,
        train: bool = False,
        shuffle=True,
        split=0.8,
    ):
        self.dataset: StagedActsDataset = dataset
        self.fields = fields
        self.split = split
        self.shuffle = shuffle
        self.train = train
        self.num_train_recs = int(len(self.dataset) * split)
        self.max_len = max_len
        self.eos_token = "<eos>"
        self.init_token = "<sos>"
        self.pad_token = "<pad>"
        self.unprocessed_data = (
            self.dataset[: self.num_train_recs]
            if self.train
            else self.dataset[self.num_train_recs :]
        )

        self.seq_data = self.process_seq_data(
            self.unprocessed_data["field_sequence"], train_fields=self.train
        ).rename(
            "field_sequence_num"
        )  # type: ignore
        self.data = self.seq_data.to_frame().merge(
            self.unprocessed_data, left_index=True, right_index=True
        )

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def process_seq_data(self, field_sequence: pd.Series[Any], train_fields):
        if train_fields:
            self._train_fields(field_sequence=field_sequence)
        act_seqs = field_sequence.apply(lambda x: x[: self.max_len - 2])
        numer_act_seqs = act_seqs.apply(  # type: ignore
            self.add_start_stop_and_numericalize_and_pad,
            args=(
                self.max_len,
                self.fields,
                self.init_token,
                self.eos_token,
                self.pad_token,
            ),
        )
        return numer_act_seqs

    def _train_fields(self, field_sequence: pd.Series):
        """Trains the torch text field tokenizer representing each independent category.
        We need to keep the tokenizers independent for the categorical levels, so that the associated embedding will remain independent"""
        for idx, field in enumerate(self.fields):
            field.build_vocab(
                [
                    [act[idx] for act in actlist if not pd.isna(act[idx])]
                    for actlist in field_sequence
                ],
                specials=["<pad>"],
                min_freq=field.min_frequency,
            )

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def numericalize(act_seq, fields: List[TTFieldWrapper]):
        """ convert the independent category values into the integer representation"""
        return [
            [field.vocab.stoi[act[idx]] for idx, field in enumerate(fields)]
            for act in act_seq
        ]

    @staticmethod
    def add_start_stop_and_numericalize_and_pad(
        act_seq, max_len, fields, init_token, eos_token, pad_token
    ):
        """ add the start and stop groups to each sequence, and pad out each seq to the max len specified """
        if pd.isna(act_seq[0][0]):
            act_seq = [
                [field.vocab.stoi[init_token] for field in fields],
                [field.vocab.stoi[eos_token] for field in fields],
            ]
            # assume type, st, lvl, rg have pad tokens that reference the same numericalized value
            act_seq.extend(
                [[fields[0].vocab.stoi[pad_token]] * 4] * (max_len - len(act_seq))
            )
            return act_seq

        else:
            act_seq.insert(0, [init_token] * len(act_seq[0]))
            act_seq.append([eos_token] * len(act_seq[0]))
            act_seq.extend([[pad_token] * 4] * (max_len - len(act_seq)))
            return StagedActsDatasetProcessor.numericalize(act_seq, fields)


if __name__ == "__main__":
    pass
    # sa = StagedActsDataset("staged_activities.csv", [Textify(feature_cols)])

    # trn, tst = torch.utils.data.random_split(sa, [l := int(len(sa) * 0.8), len(sa) - l])  # type: ignore

    # eos_token = "<eos>"
    # init_token = "<sos>"
    # pad_token = "<pad>"

    # TYPE = TTFieldWrapper(
    #     sequential=False,
    #     unk_token="<unk>",
    #     lower=True,
    #     eos_token=eos_token,
    #     init_token=init_token,
    #     pad_token=pad_token,
    #     min_frequency=100,
    # )

    # SUBTYPE = TTFieldWrapper(
    #     sequential=False,
    #     unk_token="<unk>",
    #     lower=True,
    #     eos_token=eos_token,
    #     init_token=init_token,
    #     pad_token=pad_token,
    #     min_frequency=15,
    # )

    # LVL = TTFieldWrapper(
    #     sequential=False,
    #     unk_token="<unk>",
    #     lower=True,
    #     eos_token=eos_token,
    #     init_token=init_token,
    #     pad_token=pad_token,
    #     min_frequency=50,
    # )

    # RESPGROUP = TTFieldWrapper(
    #     min_frequency=7,
    #     sequential=False,
    #     unk_token="<unk>",
    #     lower=True,
    #     eos_token=eos_token,
    #     init_token=init_token,
    #     pad_token=pad_token,
    # )

    # a = StagedActsDatasetProcessor(
    #     sa, [TYPE, SUBTYPE, LVL, RESPGROUP], max_len=6, train=True, shuffle=True
    # )
