from typing import Tuple, Dict, Any
from .base import BaseBoxModel
from .max_margin_models import MaxMarginConditionalClassificationModel
from typing import Optional
from allennlp.models import Model
from allennlp.training.metrics import Average
from boxes.box_wrapper import SigmoidBoxTensor, BoxTensor
from allennlp.modules.token_embedders import Embedding
import torch.nn as nn
import torch
import numpy as np
from ..metrics import HitsAt10


@Model.register('transE-model')
class TransEModel(MaxMarginConditionalClassificationModel):
    def get_r_vol(self) -> torch.Tensor:
        with torch.no_grad():
            v = 0

        return v
    def get_h_vol(self) -> torch.Tensor:
        with torch.no_grad():
            v = 0

        return v
    def get_t_vol(self) -> torch.Tensor:
        with torch.no_grad():
            v = 0

        return v
    def create_embeddings_layer(self, num_entities: int, num_relations: int,
                                embedding_dim: int, single_vec: bool,
                                entities_init_interval_center: float,
                                entities_init_interval_delta: float,
                                relations_init_interval_center: float,
                                relations_init_interval_delta: float) -> None:
        self.h = nn.Embedding(
            num_embeddings=num_entities,
            embedding_dim=embedding_dim)

        if not single_vec:
            self.t = nn.Embedding(
                num_embeddings=num_entities,
                embedding_dim=embedding_dim)
        else:
            self.t = self.h

        self.r = nn.Embedding(
            num_embeddings=num_relations,
            embedding_dim=embedding_dim)
        nn.init.xavier_uniform_(self.h.weight.data)
        nn.init.xavier_uniform_(self.t.weight.data)
        nn.init.xavier_uniform_(self.r.weight.data)

        self.appropriate_emb = {
            'p_h': self.h,
            'n_h': self.h,
            'h': self.h,
            'tr_h': self.h,
            'hr_e': self.h,
            'p_t': self.t,
            'n_t': self.t,
            't': self.t,
            'hr_t': self.t,
            'tr_e': self.t,
            'p_r': self.r,
            'n_r': self.r,
            'r': self.r,
            'hr_r': self.r,
            'tr_r': self.r,
            'label': (lambda x: x)
        }


    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int,
            single_vec: bool = True,
            regularization_weight: float = 0.0,
            number_of_negative_samples: int = 3,
            margin: float = 0.0,
            vocab: Optional[None] = None,
            debug: bool = False
            # we don't need vocab but some api relies on its presence as an argument
    ) -> None:
        super().__init__(num_entities, num_relations, embedding_dim, 
                         regularization_weight=regularization_weight,
                         number_of_negative_samples=number_of_negative_samples)
        self.create_embeddings_layer(num_entities,
                                     num_relations,
                                     embedding_dim, single_vec, 0.5, 0.5, 0.5, 0.5)
        self.loss_f: torch.nn.modules._Loss = torch.nn.MarginRankingLoss(  # type: ignore
            margin=margin, reduction='mean')
        self.margin = margin

    def get_scores(self,
                   embeddings: Dict) -> Tuple[torch.Tensor, torch.Tensor]:

        p_s = self._get_triple_score(embeddings['p_h'], embeddings['p_t'],
                                     embeddings['p_r'])

        n_s = self._get_triple_score(embeddings['n_h'], embeddings['n_t'],
                                     embeddings['n_r'])
        if self.regularization_weight > 0:
            self.reg_loss = self.get_regularization_penalty_vector(
                                                    embeddings['p_h'], 
                                                    embeddings['p_t'],
                                                    embeddings['p_r'])
        return (p_s, n_s)

    def get_regularization_penalty(self):
        return self.regularization_weight*self.reg_loss

    def _get_triple_score(self, head: torch.Tensor, tail: torch.Tensor,
                          relation: torch.Tensor) -> torch.Tensor:
        """ Gets score using three way intersection

        We do not need to worry about the dimentions of the boxes. If
            it can sensibly broadcast it will.
        """
        return -torch.norm(head + relation - tail, p='fro', dim=1) 


    def get_loss(self, scores: Tuple[torch.Tensor, torch.Tensor],
                 label: torch.Tensor) -> torch.Tensor:
        # max margin loss expects label to be float
        label = label.to(scores[0].dtype)
        return self.loss_f(*scores, label) + self.regularization_weight*self.reg_loss

    def get_regularization_penalty_vector(self, h, t, r):
        regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
        return regul

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if self.is_eval():
            metrics = self.threshold_with_f1.get_metric(reset)
        else:
            # metrics = self.train_f1.get_metric(reset)
            metrics = {}
            metrics[
                'regularization_loss'] = self.reg_loss.item()

        return metrics

@Model.register('complex-model')
class ComplexModel(TransEModel):
    def create_embeddings_layer(self, num_entities: int, num_relations: int,
                                embedding_dim: int, single_vec: bool,
                                entities_init_interval_center: float,
                                entities_init_interval_delta: float,
                                relations_init_interval_center: float,
                                relations_init_interval_delta: float) -> None:
        self.h_re = nn.Embedding(
            num_embeddings=num_entities,
            embedding_dim=embedding_dim)
        self.h_im = nn.Embedding(
            num_embeddings=num_entities,
            embedding_dim=embedding_dim)

        if not single_vec:
            self.t_re = nn.Embedding(
                num_embeddings=num_entities,
                embedding_dim=embedding_dim)
            self.t_im = nn.Embedding(
                num_embeddings=num_entities,
                embedding_dim=embedding_dim)
        else:
            self.t_re = self.h_re
            self.t_im = self.t_im

        self.r_re = nn.Embedding(
            num_embeddings=num_relations,
            embedding_dim=embedding_dim)
        self.r_im = nn.Embedding(
            num_embeddings=num_relations,
            embedding_dim=embedding_dim)

        nn.init.xavier_uniform_(self.h_re.weight.data)
        nn.init.xavier_uniform_(self.h_im.weight.data)
        nn.init.xavier_uniform_(self.t_re.weight.data)
        nn.init.xavier_uniform_(self.t_im.weight.data)
        nn.init.xavier_uniform_(self.r_re.weight.data)
        nn.init.xavier_uniform_(self.r_im.weight.data)

    def __init__(
           self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int,
            single_vec: bool = True,
            regularization_weight: float = 0.0,
            number_of_negative_samples: int = 3,
            margin: float = 0.0,
            vocab: Optional[None] = None,
            debug: bool = False
            # we don't need vocab but some api relies on its presence as an argument
    
            # we don't need vocab but some api relies on its presence as an argument
    ) -> None:
        super().__init__(num_entities, num_relations, embedding_dim, 
                         regularization_weight=regularization_weight,
                         number_of_negative_samples=number_of_negative_samples,
                         )
        self.create_embeddings_layer(num_entities,
                                     num_relations,
                                     embedding_dim, single_vec, 0.5, 0.5, 0.5, 0.5)
        self.loss_f: torch.nn.modules._Loss = torch.nn.MarginRankingLoss(  # type: ignore
            margin=margin, reduction='mean')
        self.margin = margin

    def _get_triple_score(self, h_re, h_im, t_re, t_im, r_re, r_im) -> torch.Tensor:
        """ Gets score using three way intersection

        We do not need to worry about the dimentions of the boxes. If
            it can sensibly broadcast it will.
        """
        return torch.sum(
            h_re * t_re * r_re
            + h_im * t_im * r_re
            + h_re * t_im * r_im
            - h_im * t_re * r_im,
            -1
            )

    def get_scores(self,
                   embeddings: Dict) -> Tuple[torch.Tensor, torch.Tensor]:

        p_s = self._get_triple_score(embeddings['p_h_re'],
                                     embeddings['p_h_im'],
                                     embeddings['p_t_re'],
                                     embeddings['p_t_im'],
                                     embeddings['p_r_re'],
                                     embeddings['p_r_im'])


        n_s = self._get_triple_score(embeddings['n_h_re'],
                                     embeddings['n_h_im'],
                                     embeddings['n_t_re'],
                                     embeddings['n_t_im'],
                                     embeddings['n_r_re'],
                                     embeddings['n_r_im'])
        if self.regularization_weight > 0:
            self.reg_loss = self.get_regularization_penalty_vector(
                                                    embeddings['p_h_re'],
                                                    embeddings['p_h_im'],
                                                    embeddings['p_t_re'],
                                                    embeddings['p_t_im'],
                                                    embeddings['p_r_re'],
                                                    embeddings['p_r_im'])

        return (p_s, n_s)

    def get_regularization_penalty_vector(self, h_re, h_im, t_re, t_im, r_re, r_im):
        regul = (torch.mean(h_re ** 2) +
                 torch.mean(h_im ** 2) +
                 torch.mean(t_re ** 2) +
                 torch.mean(t_im ** 2) +
                 torch.mean(r_re ** 2) +
                 torch.mean(r_im ** 2)) / 6
        return regul

    def get_box_embeddings_training(  # type:ignore
            self,
            p_h: torch.Tensor,
            p_r: torch.Tensor,
            p_t: torch.Tensor,  # type:ignore
            n_h: torch.Tensor,
            n_r: torch.Tensor,
            n_t: torch.Tensor,
            **kwargs) -> Dict[str, BoxTensor]:  # type: ignore
        return {
            'p_h_re': self.h_re(p_h),
            'n_h_re': self.h_re(n_h),
            'p_h_im': self.h_im(p_h),
            'n_h_im': self.h_im(n_h),
            'p_t_re': self.t_re(p_t),
            'n_t_re': self.t_re(n_t),
            'p_t_im': self.t_im(p_t),
            'n_t_im': self.t_im(n_t),
            'p_r_re': self.r_re(p_r),
            'n_r_re': self.r_re(n_r),
            'p_r_im': self.r_im(p_r),
            'n_r_im': self.r_im(n_r)
        }

    def get_box_embeddings_val(  # type:ignore
            self,
            h: torch.Tensor,
            r: torch.Tensor,
            t: torch.Tensor,
            label : torch.Tensor
            ) -> Dict[str, BoxTensor]:  # type: ignore
        return {
            'h_re': self.h_re(h),
            'h_im': self.h_im(h),
            't_re': self.t_re(t),
            't_im': self.t_im(t),
            'r_re': self.r_re(r),
            'r_im': self.r_im(r),
            'label': label
        }

    def get_ranks(self, embeddings: Dict[str, BoxTensor]) -> Any:
        s = self._get_triple_score(embeddings['h_re'],
                                   embeddings['h_im'],
                                   embeddings['t_re'],
                                   embeddings['t_im'],
                                   embeddings['r_re'],
                                   embeddings['r_im'])
        # preds = torch.stack((p_s, n_s), dim=1)  # shape = (batch, 2)
        #self.valid_f1(preds, labels)
        labels = embeddings['label']
        # upate the metrics
        self.threshold_with_f1(s, labels)

        return {}
