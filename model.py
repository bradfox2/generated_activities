""" Pytorch model that performs next sequence prediction at the intra-sequential element level using transfomer with sequence level static data
ie. ['sos','sos'] -> ['t1', 's1'] where t and s are from different categorical sets, 
but where value of  s1 is dependent on t1
"""

import itertools

import numpy as np
import torch
import torch.nn as nn
import torchtext
from tqdm import tqdm
from transformers import AdamW, DistilBertModel, DistilBertTokenizer

from data_processing import LVL, RESPGROUP, SUBTYPE, TYPE, process
from load_staged_acts import get_dat_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAModel(nn.Module):
    def __init__(self) -> None:
        super(SAModel).__init__()


trnseq, tstseq, trnstat, tststat = get_dat_data()
