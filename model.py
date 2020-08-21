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
from utils import get_field_term_weights


class IndependentCategorical(object):
    def __init__(
        self, name: str, num_levels: int, padding_idx: int, term_weights: List[Tensor]
    ) -> None:
        self.name = name
        self.num_levels = num_levels
        self.padding_idx = padding_idx
        self.term_weights: List = term_weights

    @classmethod
    def from_torchtext_field(cls, name: str, field: Field, padding_token="<pad>"):
        num_levels: int = len(field.vocab.itos)
        padding_idx: int = field.vocab.stoi[padding_token]
        term_weights: Tensor = get_field_term_weights(field)
        return IndependentCategorical(name, num_levels, padding_idx, term_weights)


class SAModelConfig(object):
    p_class_drop = 0.1
    learning_rate_decay_rate = 0.98

    def __init__(
        self,
        model_name: str,
        sequence_length: int,
        independent_categoricals: List[IndependentCategorical],
        static_data_embedding_size: int,
        freeze_static_model_weights: bool,
        categorical_embedding_dim: int,
        num_attn_heads: int,
        num_transformer_layers: int,
        **kwargs
    ) -> None:
        self.model_name = model_name
        self.sequence_length = sequence_length
        self.independent_categoricals = independent_categoricals
        self.static_data_embedding_size = static_data_embedding_size
        self.freeze_static_model_weights = freeze_static_model_weights
        self.categorical_embedding_dim = categorical_embedding_dim
        self.num_attn_heads = num_attn_heads
        self.num_transfomers_layers = num_transformer_layers

        self.num_act_cats = len(self.independent_categoricals)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return str(self.__dict__)


class LargeSAModel(SAModelConfig):
    model_name = "LargeSAModel"
    batch_sz = 32
    categorical_embedding_dim = 128
    num_attn_heads = 8
    num_transfomers_layers = 6
    static_data_embedding_size = 768

    def __init__(self, **kwargs) -> None:
        super(LargeSAModel, self).__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)


class VerySmallSAModel(SAModelConfig):
    model_name = "VerySmallSAModel"
    batch_sz = 128
    categorical_embedding_dim = 16
    num_attn_heads = 2
    num_transfomers_layers = 1
    learning_rate_decay_rate = 0.98
    static_data_embedding_size = 768

    def __init__(self, **kwargs) -> None:
        super(VerySmallSAModel, self).__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)


class SAModel(nn.Module):
    def __init__(
        self, config: SAModelConfig, device: torch.device = torch.device("cpu"),
    ) -> None:

        super(SAModel, self).__init__()
        self.sequence_length = config.sequence_length
        self.categorical_embedding_dim = config.categorical_embedding_dim
        self.num_attn_heads = config.num_attn_heads
        self.num_transformer_layers = config.num_transformer_layers
        self.independent_categoricals = config.independent_categoricals
        self.p_class_drop = config.p_class_drop
        self.device = device
        self.num_independent_categoricals = len(config.independent_categoricals)
        assert (
            self.categorical_embedding_dim
            * self.num_independent_categoricals
            % self.num_attn_heads
            == 0
        )
        self.transformer_dim_sz = (
            self.categorical_embedding_dim * self.num_independent_categoricals
        )
        self.weight_initrange = 0.1
        self._generate_embedding_layers()
        self._generate_classification_layers()
        self.apply(self._init_weights)

        self._generate_loss_criterion()
        self.cat_emb_layer_norm = nn.LayerNorm(
            self.categorical_embedding_dim * self.num_independent_categoricals
        )

        self.mask = None
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            self.transformer_dim_sz, self.num_attn_heads
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_decoder_layer, self.num_transformer_layers
        )
        self.classification_tnsr_drop = nn.Dropout(self.p_class_drop)
        self.optimizer = torch.optim.AdamW(self.parameters())

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

        if config.freeze_static_model_weights:
            for param in self.static_data_model.parameters():
                param.requires_grad = False

        self.static_data_embedding_size = config.static_data_embedding_size
        self.static_data_squeeze = nn.Linear(
            self.static_data_embedding_size, self.transformer_dim_sz
        )
        self.static_data_layer_norm = nn.LayerNorm(self.transformer_dim_sz)

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 5.0, config.learning_rate_decay_rate
        )
        self.static_optimizer = transformers.AdamW(
            self.static_data_model.parameters(), lr=1e-5
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
                nn.Linear(self.transformer_dim_sz, ind_cat.num_levels)
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

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

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
        ).to(self.device)

        assert self._pad_tokens_identical()
        tgt_pad_idx = self.independent_categoricals[0].padding_idx

        static_data_embedding = (
            self.static_data_model(**static_data)[0].mean(1).unsqueeze(0)
        )

        # ensure batch size of static data same as seqential data
        assert list(static_data_embedding.shape[:-1]) == [1, data.shape[1]]

        static_data_embedding_squeeze = self.static_data_squeeze(static_data_embedding)

        cat_embeddings_list = []
        for idx, embedding in enumerate(self.cat_embeddings):
            cat_embeddings_list.append(embedding(data[..., idx]))

        cats_combined_embedding = torch.cat(cat_embeddings_list, dim=2)
        # cat_embs_nrm = self.cat_emb_layer_norm(cats_combined_embedding)

        if self.training:
            tgt_key_padding_mask = data == tgt_pad_idx
            # dims = (target sequence length x batch size)
            tgt_key_padding_mask = tgt_key_padding_mask.permute(1, 0, 2)[:, :, 0]
        else:
            tgt_key_padding_mask = None
            # self.mask = None

        tfmr_out = self.transformer_decoder(
            cats_combined_embedding,
            memory=static_data_embedding_squeeze,
            tgt_mask=self.mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        tfmr_out = self.classification_tnsr_drop(tfmr_out)
        classification_layer_outputs = []
        for classification_layer in self.cat_linear_classifiers:
            classification_layer_outputs.append(classification_layer(tfmr_out))
        return classification_layer_outputs

    def learn(self, data: Tensor, static_data: Tensor, targets: Tensor):
        self.optimizer.zero_grad()
        self.static_optimizer.zero_grad()

        data = data
        targets = targets

        preds = self.forward(data, static_data)

        loss = self.loss(preds, targets)

        loss.backward()
        self.optimizer.step()
        self.static_optimizer.step()

        return loss.item()
