""" Pytorch model that performs next sequence prediction at the intra-sequential element level using transformer with sequence level static data
ie. ['sos','sos'] -> ['t1', 's1'] where t and s are from different categorical sets, 
but where value of  s1 is dependent on t1
"""

from typing import List

import torch
import torch.nn as nn
import transformers
from torch.tensor import Tensor
from torchtext.data.field import Field
from transformers import DistilBertModel
import math
from utils import get_field_term_weights


class IndependentCategorical(object):
    def __init__(
        self, name: str, num_levels: int, padding_idx: int, term_weights: List[Tensor],
    ) -> None:
        """ independent categorical used to create embedding and classification layers"""
        self.name = name
        self.num_levels = num_levels
        self.padding_idx = padding_idx
        self.term_weights: List = term_weights

    @classmethod
    def from_torchtext_field(
        cls, name: str, field: Field, total_num_samples: int, padding_token="<pad>"
    ):
        num_levels: int = len(field.vocab.itos)
        padding_idx: int = field.vocab.stoi[padding_token]
        term_weights: Tensor = get_field_term_weights(field, total_num_samples)
        return IndependentCategorical(name, num_levels, padding_idx, term_weights)


class SAModelConfig(object):
    pass


class ActClassifierHead(nn.Module):
    def __init__(self, n_embd, p_drop, n_classes) -> None:
        super(ActClassifierHead, self).__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(p_drop),
        )
        self.final_class = nn.Linear(n_embd, n_classes)

    def forward(self, x):
        return self.final_class(self.mlp(self.ln1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class SAModel(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        categorical_embedding_dim: int,
        num_attn_heads: int,
        num_hidden: int,
        num_transformer_layers: int,
        learning_rate: float,
        independent_categoricals: List[IndependentCategorical],
        freeze_static_model_weights: bool,
        warmup_steps: int,
        total_steps: int,
        device,
        static_data_embedding_size: int = 768,
        dropout: float = 0.2,
        grad_norm_clip: float = 1.0,
    ) -> None:

        super(SAModel, self).__init__()
        self.sequence_length = sequence_length
        self.categorical_embedding_dim = categorical_embedding_dim
        self.num_attn_heads = num_attn_heads
        self.num_hidden = num_hidden
        self.num_transformer_layers = num_transformer_layers
        self.independent_categoricals = independent_categoricals
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.num_independent_categoricals = len(independent_categoricals)
        assert (
            self.categorical_embedding_dim
            * self.num_independent_categoricals
            % self.num_attn_heads
            == 0
        )
        self.transformer_dim_sz = (
            categorical_embedding_dim * self.num_independent_categoricals
        )
        self.dropout = dropout
        self.device = device
        self.weight_initrange = 0.1
        self._generate_embedding_layers()
        self._generate_classification_layers()
        self._generate_loss_criterion()
        # self._init_weights()
        self.cat_emb_layer_norm = nn.LayerNorm(
            self.categorical_embedding_dim * self.num_independent_categoricals
        )
        self.grad_norm_clip = grad_norm_clip
        self.mask = None
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            self.transformer_dim_sz, self.num_attn_heads, self.num_hidden, self.dropout,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_decoder_layer, self.num_transformer_layers
        )
        self.classification_tnsr_drop = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(
            self.num_independent_categoricals * self.categorical_embedding_dim, dropout
        )

        try:
            self.static_data_model = DistilBertModel.from_pretrained(
                "./distilbert_weights"
            )
        except:
            try:
                self.static_data_model = DistilBertModel.from_pretrained(
                    "distilbert-base-uncased"
                )
            except Exception as e:
                raise (e)

        if freeze_static_model_weights:
            for param in self.static_data_model.parameters():
                param.requires_grad = False

        self.static_data_embedding_size = static_data_embedding_size
        self.static_data_squeeze = nn.Linear(
            self.static_data_embedding_size, self.transformer_dim_sz
        )
        self.static_data_layer_norm = nn.LayerNorm(self.transformer_dim_sz)

        self.optimizer = torch.optim.AdamW(self.parameters(), self.learning_rate)

        assert self._pad_tokens_identical()
        self.tgt_pad_idx = self.independent_categoricals[0].padding_idx
        self.scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            self.optimizer, self.warmup_steps, self.total_steps, 6
        )

    def _pad_tokens_identical(self):
        pts = {ind_cat.padding_idx for ind_cat in self.independent_categoricals}
        if len(pts) > 1:
            raise ValueError(
                "Padding index values should be the same for all independent categoricals to support loss masking."
            )
        return True

    def _generate_loss_criterion(self):
        " generates an instance of a loss criterion for each independent categorical"
        self.loss_criteria = [
            nn.CrossEntropyLoss(
                ignore_index=ind_cat.padding_idx, weight=ind_cat.term_weights
            ).to(self.device)
            for ind_cat in self.independent_categoricals
        ]

    def loss(self, preds: List[Tensor], targets: Tensor):
        """ Get loss b/w preds and targets using the list of crierion"""
        loss = None

        for idx, pred in enumerate(preds):
            pred_cat_num_levels = self.independent_categoricals[idx].num_levels
            pred = pred.reshape(
                pred.numel() // pred_cat_num_levels, pred_cat_num_levels
            )
            target = targets[..., idx].flatten()
            crit = self.loss_criteria[idx]
            loss = crit(pred, target) if not loss else crit(pred, target) + loss
        return loss

    def _generate_classification_layers(self):
        " Generates a linear classification layer for each independent categorical."
        self.cat_linear_classifiers = nn.ModuleList(
            [
                ActClassifierHead(
                    self.transformer_dim_sz, self.dropout, ind_cat.num_levels
                )
                for ind_cat in self.independent_categoricals
            ]
        )

    def _generate_embedding_layers(self):
        " Generate embedding layers for each independent categorical."
        # TODO: allow dynamic embedding dim based on number of levels in each category
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    ind_cat.num_levels,
                    self.categorical_embedding_dim,
                    padding_idx=ind_cat.padding_idx,
                )
                for ind_cat in self.independent_categoricals
            ]
        )

    def _init_weights(self):
        " Initialize weights appropriate for transformer for each linear and embedding layer."
        for layer in self.cat_embeddings:
            layer.weight.data.uniform_(-self.weight_initrange, self.weight_initrange)

        for layer in self.cat_linear_classifiers:
            layer.weight.data.uniform_(-self.weight_initrange, self.weight_initrange)
            layer.bias.data.zero_()

    def _generate_square_target_mask(self, seq_len):
        """ Generates a top right triangle square mask of the target sequence.  
        Prevents attending to targets that only exist forward in time. """
        tgt_mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        tgt_mask = (
            tgt_mask.float()
            .masked_fill(tgt_mask == 0, float("-inf"))
            .masked_fill(tgt_mask == 1, float(0.0))
        )
        return tgt_mask

    def forward(self, data: Tensor, static_data: Tensor):
        self.mask = (
            self._generate_square_target_mask(self.sequence_length)
            if self.mask is None
            else self.mask
        )  # .to(self.device)

        # combine all DB word vectors emitted into a single timestep feature vector
        static_data_embedding = (
            self.static_data_model(**static_data)[0].mean(1).unsqueeze(0)
        )

        # ensure batch size of static data same as seqential data
        assert list(static_data_embedding.shape[:-1]) == [1, data.shape[1]]

        # static_data_embedding = self.static_data_squeeze(static_data_embedding)

        cat_embeddings_list = []
        for idx, embedding in enumerate(self.cat_embeddings):
            cat_embeddings_list.append(embedding(data[..., idx]))

        cats_combined_embedding = torch.cat(cat_embeddings_list, dim=2)  # * math.sqrt(
        # self.transformer_dim_sz)

        # cats_combined_embedding = self.pos_encoder(cats_combined_embedding)

        tgt_key_pad_mask = data == self.tgt_pad_idx
        tgt_key_pad_mask = tgt_key_pad_mask.permute(1, 0, 2)[
            :, :, 0
        ]  # (target sequence length x batch size)

        tfmr_out = self.transformer_decoder(
            cats_combined_embedding,
            memory=static_data_embedding,
            tgt_mask=self.mask.to(self.device),
            tgt_key_padding_mask=tgt_key_pad_mask,  # .to(self.device),
        )

        tfmr_out = self.classification_tnsr_drop(tfmr_out)
        classification_layer_outputs = []
        for classification_layer in self.cat_linear_classifiers:
            classification_layer_outputs.append(classification_layer(tfmr_out))
        return classification_layer_outputs

    def learn(self, data: Tensor, static_data: Tensor, targets: Tensor):
        self.optimizer.zero_grad()

        preds = self.forward(data, static_data)

        loss = self.loss(preds, targets)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm_clip)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()
