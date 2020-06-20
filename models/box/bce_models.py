from typing import Tuple, Dict, Any, Union
from .base import BaseBoxModel
from .max_margin_models import MaxMarginBoxModel, MaxMarginConditionalModel, MaxMarginConditionalClassificationModel
from typing import Optional
from allennlp.models import Model
from allennlp.training.metrics import Average
from boxes.box_wrapper import SigmoidBoxTensor, BoxTensor
from boxes.modules import BoxEmbedding
from boxes.utils import log1mexp
from allennlp.modules.token_embedders import Embedding
import torch
import numpy as np
from ..metrics import HitsAt10, F1WithThreshold
from allennlp.training.metrics import F1Measure, FBetaMeasure


@Model.register('BCE-box-model')
class BCEBoxModel(MaxMarginConditionalModel):
    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 embedding_dim: int,
                 box_type: str = 'SigmoidBoxTensor',
                 single_box: bool = False,
                 softbox_temp: float = 10.,
                 number_of_negative_samples: int = 0,
                 debug: bool = False,
                 regularization_weight: float = 0,
                 init_interval_center: float = 0.25,
                 init_interval_delta: float = 0.1) -> None:
        super().__init__(
            num_entities,
            num_relations,
            embedding_dim,
            box_type=box_type,
            single_box=single_box,
            softbox_temp=softbox_temp,
            margin=0.0,
            number_of_negative_samples=number_of_negative_samples,
            debug=debug,
            regularization_weight=regularization_weight,
            init_interval_center=init_interval_center,
            init_interval_delta=init_interval_delta)

        self.loss_f = torch.nn.NLLLoss(reduction='mean')

    def get_box_embeddings_training(  # type:ignore
            self,
            h: torch.Tensor,
            r: torch.Tensor,
            t: torch.Tensor,  # type:ignore
            label: torch.Tensor,
            **kwargs) -> Dict[str, BoxTensor]:  # type: ignore

        return {
            'h': self.h(h),
            't': self.t(t),
            'r': self.r(r),
            'label': label,
        }

    def get_scores(self, embeddings: Dict) -> torch.Tensor:
        p = self._get_triple_score(embeddings['h'], embeddings['t'],
                                   embeddings['r'])

        return p

    def get_loss(self, scores: torch.Tensor,
                 label: torch.Tensor) -> torch.Tensor:
        log_p = scores
        log1mp = log1mexp(log_p)
        logits = torch.stack([log1mp, log_p], dim=-1)
        loss = self.loss_f(logits, label) + self.get_regularization_penalty()

        return loss

    def batch_with_negative_samples(self, **kwargs) -> Dict[str, torch.Tensor]:
        if self.number_of_negative_samples <= 0:
            return kwargs
        head_name, head = self.get_expected_head(kwargs)
        tail_name, tail = self.get_expected_tail(kwargs)
        rel_name, rel = self.get_expected_relation(kwargs)
        label = kwargs.pop('label', None)

        if label is None:
            raise ValueError("Training without labels!")
        # create the tensors for negatives
        #size = self.get_negaive_sample_tensorsize(head.size())
        # for Classification model, we will do it inplace
        multiplier = int(self.number_of_negative_samples / 2)
        size = head.size()[-1]
        head = self.repeat(head, 2 * multiplier + 1)
        tail = self.repeat(tail, 2 * multiplier + 1)
        rel = self.repeat(rel, 2 * multiplier + 1)
        label = self.repeat(label, 2 * multiplier + 1)

        # fill in the random
        self.fill_random_entities_(head[size:size + size * multiplier])
        label[size:size * multiplier + size] = 0
        self.fill_random_entities_(tail[size * (1 + multiplier):])
        label[size * (1 + multiplier):] = 0

        return {'h': head, 't': tail, 'r': rel, 'label': label}


@Model.register('BCE-classification-box-model')
class BCEBoxClassificationModel(BCEBoxModel):
    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 embedding_dim: int,
                 box_type: str = 'SigmoidBoxTensor',
                 single_box: bool = False,
                 softbox_temp: float = 10.,
                 number_of_negative_samples: int = 0,
                 debug: bool = False,
                 regularization_weight: float = 0,
                 init_interval_center: float = 0.25,
                 init_interval_delta: float = 0.1) -> None:
        super().__init__(
            num_entities,
            num_relations,
            embedding_dim,
            box_type=box_type,
            single_box=single_box,
            softbox_temp=softbox_temp,
            number_of_negative_samples=number_of_negative_samples,
            debug=debug,
            regularization_weight=regularization_weight,
            init_interval_center=init_interval_center,
            init_interval_delta=init_interval_delta)
        self.train_f1 = FBetaMeasure(average='micro')
        #self.valid_f1 = FBetaMeasure(average='micro')
        self.threshold_with_f1 = F1WithThreshold(flip_sign=True)

        self.loss_f = torch.nn.NLLLoss(reduction='mean')
        self.istest = False
        self.test_threshold = None
        #self.test_f1 = FBetaMeasure(average='macro')
        self.test_f1 = F1Measure(positive_label=1)

    def is_test(self) -> bool:
        if (not self.is_eval()) and self.test:
            raise RuntimeError("test flag is true but eval is false")

        return self.is_eval() and self.istest

    def test(self) -> None:
        if not self.is_eval():
            raise RuntimeError("test flag is true but eval is false")
        self.istest = True

    def get_box_embeddings_val(self, h: torch.Tensor, t: torch.Tensor,
                               r: torch.Tensor,
                               label: torch.Tensor) -> Dict[str, BoxTensor]:

        return BaseBoxModel.get_box_embeddings_val(
            self, h=h, t=t, r=r, label=label)

    def get_loss(self, scores: torch.Tensor,
                 label: torch.Tensor) -> torch.Tensor:
        log_p = scores
        log1mp = log1mexp(log_p)
        logits = torch.stack([log1mp, log_p], dim=-1)
        loss = self.loss_f(logits, label) + self.get_regularization_penalty()

        if not self.is_eval():
            with torch.no_grad():
                self.train_f1(logits, label)

        return loss

    def get_ranks(self, embeddings: Dict[str, BoxTensor]) -> Any:
        if self.is_test():
            return self.get_test(embeddings)

        s = self._get_triple_score(embeddings['h'], embeddings['t'],
                                   embeddings['r'])
        # preds = torch.stack((p_s, n_s), dim=1)  # shape = (batch, 2)
        #self.valid_f1(preds, labels)
        labels = embeddings['label']
        # upate the metrics
        self.threshold_with_f1(s, labels)

        return {}

    # def get_test(self, embeddings: Dict[str, BoxTensor]) -> Any:
    #    # breakpoint()

    #    if self.test_threshold is None:
    #        raise RuntimeError("test_threshold should be set")
    #    log_p = self._get_triple_score(embeddings['h'], embeddings['t'],
    #                                   embeddings['r'])
    #    labels = embeddings['label']
    #    log1mp = log1mexp(log_p)
    #    logits = torch.stack([log1mp, log_p], dim=-1)
    #    self.test_f1(logits, labels)


#
#        return {}

    def get_test(self, embeddings: Dict[str, BoxTensor]) -> Any:
        if self.test_threshold is None:
            raise RuntimeError("test_threshold should be set")
        s = self._get_triple_score(embeddings['h'], embeddings['t'],
                                   embeddings['r'])
        labels = embeddings['label']
        pos_prediction = (s > self.test_threshold).float()
        neg_prediction = 1.0 - pos_prediction
        predictions = torch.stack((neg_prediction, pos_prediction), -1)
        self.test_f1(predictions, labels)

        return {}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if self.is_eval():
            if not self.istest:
                metrics = self.threshold_with_f1.get_metric(reset)
            else:
                p, r, f = self.test_f1.get_metric(reset)
                metrics = {'precision': p, 'recall': r, 'fscore': f}

        else:
            metrics = self.train_f1.get_metric(reset)
            metrics[
                'regularization_loss'] = self.regularization_loss.get_metric(
                    reset)

        return metrics


@Model.register('BCE-classification-split-neg-box-model')
class BCEBoxClassificationSplitNegModel(BCEBoxClassificationModel):
    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 embedding_dim: int,
                 box_type: str = 'SigmoidBoxTensor',
                 single_box: bool = False,
                 softbox_temp: float = 10.,
                 number_of_negative_samples_head: int = 0,
                 number_of_negative_samples_tail: int = 0,
                 debug: bool = False,
                 regularization_weight: float = 0,
                 init_interval_center: float = 0.25,
                 init_interval_delta: float = 0.1) -> None:
        super().__init__(
            num_entities,
            num_relations,
            embedding_dim,
            box_type=box_type,
            single_box=single_box,
            softbox_temp=softbox_temp,
            number_of_negative_samples=number_of_negative_samples_head +
            number_of_negative_samples_tail,
            debug=debug,
            regularization_weight=regularization_weight,
            init_interval_center=init_interval_center,
            init_interval_delta=init_interval_delta)
        self.number_of_negative_samples_head = number_of_negative_samples_head
        self.number_of_negative_samples_tail = number_of_negative_samples_tail

    def batch_with_negative_samples(self, **kwargs) -> Dict[str, torch.Tensor]:
        if self.number_of_negative_samples <= 0:
            return kwargs
        head_name, head = self.get_expected_head(kwargs)
        tail_name, tail = self.get_expected_tail(kwargs)
        rel_name, rel = self.get_expected_relation(kwargs)
        label = kwargs.pop('label', None)

        if label is None:
            raise ValueError("Training without labels!")
        # create the tensors for negatives
        #size = self.get_negaive_sample_tensorsize(head.size())
        # for Classification model, we will do it inplace
        multiplier = int(self.number_of_negative_samples)
        size = head.size()[-1]
        head = self.repeat(head, multiplier + 1)
        tail = self.repeat(tail, multiplier + 1)
        rel = self.repeat(rel, multiplier + 1)
        label = self.repeat(label, multiplier + 1)

        # fill in the random
        head_multiplier = int(self.number_of_negative_samples_head)
        self.fill_random_entities_(head[size:size + size * head_multiplier])
        tail_multiplier = int(self.number_of_negative_samples_tail)

        if tail_multiplier > 0:
            self.fill_random_entities_(
                tail[size + size * head_multiplier:size +
                     size * head_multiplier + size * tail_multiplier])
        label[size:size * multiplier + size] = 0

        return {'h': head, 't': tail, 'r': rel, 'label': label}


@Model.register('BCE-classification-split-neg-vol-penalty-box-model')
class BCEBoxClassificationSplitNegVolPenaltyModel(
        BCEBoxClassificationSplitNegModel):
    def get_regularization_penalty(self) -> Union[float, torch.Tensor]:

        if self.is_eval():
            return 0.0

        if self.regularization_weight > 0:
            vols = torch.exp(self.h.get_volumes(temp=self.softbox_temp))
            min_vol = 0.1
            # don't penalize if box has very less vol
            #deviation = (vols - 0.5)**2
            #large_mask = (vols > (0.01)**self.embedding_dim)
            small_mask = (vols < min_vol)
            vols = vols[small_mask]
            diff = min_vol - vols
            #penalty = self.regularization_weight * torch.sum(vols)
            penalty = self.regularization_weight * torch.sum(diff)
            # track the reg loss
            self.regularization_loss(penalty.item())

            return penalty
        else:
            return 0.0
