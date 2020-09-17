"""loads long dataset of staged acts with the cr details, and makes staged act sequences grouped by crs"""

from typing import List

import pandas as pd

from utils import set_seed

set_seed(0)

staged_activity_fields = [
    "SA_WRK_TYPE",
    "SA_SUB_TYPE",
    "SA_CAP_LVL",
    "SA_WRK_RESP_ORG_UNIT",
]


def create_act_seqs(df, seq_field_names, group_column_name="CR_CD"):
    """Create a list of list of column values specificed in seq_field_names, as grouped by group_column_name"""
    df["field_sequence"] = df[staged_activity_fields].values.tolist()
    act_seqs = df.groupby(group_column_name)["field_sequence"].apply(list)
    return act_seqs


feature_cols = [
    "LI_QCLS_CD",
    "LI_FAIL_CD",
    "CAP_CLASS",
    "RESPONSIBLE_GROUP",
    "DESCR",
    "LEADER_COMMENT",
]


def textify_feature_data(cr_data, feature_cols):
    cr_data["TEXT"] = ""
    for col in feature_cols:
        cr_data["TEXT"] += f" [{col}] " + cr_data.get(col, "[SEP]").astype(str)

    return cr_data


def prep_feature_data(csv_path: str, feature_cols: List) -> pd.DataFrame:
    """get data from csv and textify it"""
    cr_data = pd.read_csv(
        csv_path,
        dtype={k: object for k in staged_activity_fields},
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
    cr_data = cr_data.sample(frac=1, random_state=1)

    return textify_feature_data(cr_data, feature_cols)


def get_dat_data(split_frac: float, feature_cols: List):
    """ get data from the csv, extract columns and split into test/train"""
    cr_data = prep_feature_data("staged_activites.csv", feature_cols)

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


if __name__ == "__main__":
    get_dat_data()
