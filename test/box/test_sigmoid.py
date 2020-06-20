from models.box.sigmoid import SigmoidBoxModel
from allennlp.common.params import Params
from allennlp.models.model import Model
import pprint


def test_SigmoidBoxModelFromParams() -> None:
    params = Params({
        "model": {
            "type": 'sigmoid-box-model',
            "num_entities": 100,
            "num_relations": 12,
            "embedding_dim": 50
        }
    })
    print('Creating model using following config')
    pprint.pprint(params)
    model = Model.from_params(params['model'])
    print(model)


if __name__ == '__main__':
    test_SigmoidBoxModelFromParams()
