"""loads long dataset of staged acts with the cr details, and makes staged act sequences grouped by crs"""

from typing import Any, List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from utils import set_seed

set_seed(0)


def create_act_seqs(df, seq_field_names, group_column_name="CR_CD"):
    StagedActsDataset.create_act_seqs(df)


feature_cols = [
    "LI_QCLS_CD",
    "LI_FAIL_CD",
    "CAP_CLASS",
    "RESPONSIBLE_GROUP",
    "DESCR",
    "LEADER_COMMENT",
]


def textify_feature_data(cr_data, feature_cols):
    return StagedActsDataset.textify_feature_data(cr_data, feature_cols)


def prep_feature_data(csv_path: str, feature_cols: List) -> pd.DataFrame:
    """get data from csv and textify it"""
    cr_data = StagedActsDataset.get_data(csv_path)
    cr_data = cr_data.sample(frac=1, random_state=1)
    return textify_feature_data(cr_data, feature_cols)


def get_dat_data(file: str, split_frac: float, feature_cols: List):
    """ get data from the csv, extract columns and split into test/train"""
    cr_data = prep_feature_data(file, feature_cols)
    cr_data = textify_feature_data(cr_data, feature_cols)

    tst_data = cr_data[
        cr_data.CR_CD.isin(cr_data.sample(frac=1.0 - split_frac, random_state=1).CR_CD)
    ]

    assert len(tst_data) > 1, "Need tst data."
    trn_data = cr_data[~cr_data.CR_CD.isin(tst_data.CR_CD)]

    assert len(trn_data) > 1, "Need training data."

    trn_act_seqs = create_act_seqs(trn_data, staged_activity_fields)

    trn_static_data = trn_data[["CR_CD", "TEXT"]].drop_duplicates().set_index("CR_CD")

    trn_static_data = trn_static_data[trn_static_data.index.isin(trn_act_seqs.index)]

    trn_static_data = trn_static_data.reindex(trn_act_seqs.index)

    assert len(trn_static_data) == len(trn_act_seqs)

    tst_act_seqs = create_act_seqs(tst_data, staged_activity_fields)
    tst_static_data = tst_data[["CR_CD", "TEXT"]].drop_duplicates().set_index("CR_CD")
    tst_static_data = tst_static_data[tst_static_data.index.isin(tst_act_seqs.index)]
    tst_static_data = tst_static_data.reindex(tst_act_seqs.index)

    return trn_act_seqs, tst_act_seqs, trn_static_data, tst_static_data


class StagedActsDataset(Dataset):

    staged_activity_fields = {
        "SA_WRK_TYPE": object,
        "SA_SUB_TYPE": object,
        "SA_CAP_LVL": object,
        "SA_WRK_RESP_ORG_UNIT": object,
    }

    type_overrides = {"RESPONSIBLE_GROUP": object}

    def __init__(self, csv_path, transform=None):
        self.raw_data = self.get_data(csv_path)
        self.data = self._process_raw_data()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data.iloc[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample.to_dict()

    def _process_raw_data(self):
        data = self.raw_data.copy(deep=True).drop_duplicates().reset_index()
        feats = data.groupby("CR_CD").first()
        return pd.concat([self.create_act_seqs(data), feats], axis=1)

    @classmethod
    def get_data(cls, csv_path) -> pd.DataFrame:
        return pd.read_csv(  # type: ignore
            csv_path,
            dtype=dict(cls.staged_activity_fields, **cls.type_overrides),
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
            data["TEXT"] += f" [{col}] " + str(data.get(col, "[SEP]"))
        return data


if __name__ == "__main__":
    # get_dat_data()

    sa = StagedActsDataset("staged_activities.csv", Textify(feature_cols))
    dl = DataLoader(sa, batch_size=4, shuffle=True, num_workers=0)

    sa[0]