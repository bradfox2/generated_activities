"""loads long dataset of staged acts with the cr details, and makes staged act sequences grouped by crs"""

from os import startfile
import pandas
import numpy as np

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


def get_dat_data(split_frac: float = 0.8):
    cr_data = pandas.read_csv(
        "staged_activities.csv",
        dtype={k: object for k in staged_activity_fields},
        parse_dates=True,
    )
    cr_data = cr_data.sample(frac=1)

    tst_data = cr_data[: int((1 - split_frac) * len(cr_data))]
    trn_data = cr_data[len(tst_data) + 1 :]

    trn_act_seqs = create_act_seqs(trn_data, staged_activity_fields)
    trn_static_data = trn_data[["CR_CD", "DESCR"]].drop_duplicates().set_index("CR_CD")
    trn_static_data = trn_static_data[trn_static_data.index.isin(trn_act_seqs.index)]

    assert len(trn_static_data) == len(trn_act_seqs)

    tst_act_seqs = create_act_seqs(tst_data, staged_activity_fields)
    tst_static_data = tst_data[["CR_CD", "DESCR"]].drop_duplicates().set_index("CR_CD")
    tst_static_data = tst_static_data[tst_static_data.index.isin(tst_act_seqs.index)]

    return trn_act_seqs, tst_act_seqs, trn_static_data, tst_static_data


if __name__ == "__main__":
    get_dat_data()

