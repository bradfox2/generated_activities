"""loads long dataset of staged acts with the cr details, and makes staged act sequences grouped by crs"""

from typing import Any, Callable, List, Union

import numpy as np
import pandas as pd
import torch
import torchtext
from torch.utils.data import DataLoader, Dataset

from utils import set_seed

set_seed(0)


def create_act_seqs(df, seq_field_names, group_column_name="CR_CD"):
    """ legacy until refactor of training """
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
from typing import TypeVar

SADS = TypeVar("SADS", bound="StagedActsDataset")


class StagedActsDataset(Dataset):
    """ Base dataset for a staged activity (groups of categorical sequences) model """

    staged_activity_fields = {
        "SA_WRK_TYPE": object,
        "SA_SUB_TYPE": object,
        "SA_CAP_LVL": object,
        "SA_WRK_RESP_ORG_UNIT": object,
    }

    def __init__(self, raw_data: pd.DataFrame, transform=None):
        self.raw_data = raw_data
        self.data = self.build_sequences()
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

    def build_sequences(self):
        """ aggregates the long form sequential records, and then creates a list of the sequential categories """

        data = self.raw_data.copy(deep=True).drop_duplicates().reset_index()
        feats = data.groupby("CR_CD").first()
        d = pd.concat([self.create_act_seqs(data), feats], axis=1)
        self.data = d
        return self.data

    @classmethod
    def from_csv(cls, csv_path, transform: List[Callable]) -> SADS:
        """ Creates a Staged Activity raw dataset from a CSV File path, applies functions defined in transoforms"""
        raw_data: pd.DataFrame = pd.read_csv(  # type: ignore
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
        return StagedActsDataset(raw_data, transform)

    @classmethod
    def create_act_seqs(cls, df):
        """Create a list of list of column values specificed in seq_field_names, as grouped by group_column_name"""
        df["field_sequence"] = df[cls.staged_activity_fields].values.tolist()
        act_seqs = df.groupby("CR_CD")["field_sequence"].apply(list)
        return act_seqs

    def splits(self, trn=0.75, tst=0.15, val=0.10):
        trn_idx = int(len(self.raw_data) * trn)
        tst_idx = int(len(self.raw_data) * (trn + tst))
        trn_data = self.raw_data[:trn_idx]
        tst_data = self.raw_data[trn_idx:tst_idx]
        val_data = self.raw_data[tst_idx:]
        return [
            StagedActsDataset(d, self.transform) for d in [trn_data, tst_data, val_data]
        ]


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
        fields: List[TTFieldWrapper],
        max_len: int,
    ):
        self.fields = fields
        self.max_len = max_len
        self.eos_token = "<eos>"
        self.init_token = "<sos>"
        self.pad_token = "<pad>"

    def process_dataset_to_tokens(
        self, dataset: StagedActsDataset, train_fields: bool, sequence_col_name: str
    ) -> StagedActsDataset:
        """transforms the categorical values in the sequence column of a staged act dataset into numericals suitable for embedding
        train_fields indicated whether to train fields associated with parent processor object"""

        processed_col_name = sequence_col_name + "_num"
        numrcl_seq = self.process_series_to_tokens(
            dataset.data[sequence_col_name], train_fields
        )
        numrcl_seq.rename(processed_col_name, inplace=True)  # type: ignore
        numrcl_frame = numrcl_seq.to_frame().merge(
            dataset.data, left_index=True, right_index=True
        )
        dataset.data = numrcl_frame
        return dataset

    def process_series_to_tokens(
        self, field_sequence: pd.Series, train_fields: bool
    ) -> Union[pd.Series, pd.DataFrame]:
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


class StagedActsTrn(StagedActsDataset):
    pass


if __name__ == "__main__":
    # pass
    sa = StagedActsDataset.from_csv("staged_activities.csv", [Textify(feature_cols)])
    tr, ts, v = sa.splits()

    eos_token = "<eos>"
    init_token = "<sos>"
    pad_token = "<pad>"

    TYPE = TTFieldWrapper(
        sequential=False,
        unk_token="<unk>",
        lower=True,
        eos_token=eos_token,
        init_token=init_token,
        pad_token=pad_token,
        min_frequency=100,
    )

    SUBTYPE = TTFieldWrapper(
        sequential=False,
        unk_token="<unk>",
        lower=True,
        eos_token=eos_token,
        init_token=init_token,
        pad_token=pad_token,
        min_frequency=15,
    )

    LVL = TTFieldWrapper(
        sequential=False,
        unk_token="<unk>",
        lower=True,
        eos_token=eos_token,
        init_token=init_token,
        pad_token=pad_token,
        min_frequency=50,
    )

    RESPGROUP = TTFieldWrapper(
        min_frequency=7,
        sequential=False,
        unk_token="<unk>",
        lower=True,
        eos_token=eos_token,
        init_token=init_token,
        pad_token=pad_token,
    )

    sadp = StagedActsDatasetProcessor([TYPE, SUBTYPE, LVL, RESPGROUP], max_len=6)
    sadp.process_dataset_to_tokens(tr, True, "field_sequence")
    sadp.process_dataset_to_tokens(ts, False, "field_sequence")
    sadp.process_dataset_to_tokens(v, False, "field_sequence")
