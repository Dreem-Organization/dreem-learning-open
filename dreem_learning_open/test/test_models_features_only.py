import os
import shutil

import gc
import torch

from ..datasets.dataset import DreemDataset
from ..models.modulo_net import ModuloNet
from ..models.modulo_net.normalization import initialize_standardization_parameters
from .descriptions import three_groups_record_description
from .utils import generate_memmaps
from ..trainers.trainer import Trainer

features_memmap = {}
features_memmap['signals'] = []
features_memmap['features'] = [
    {
        'name': 'fft',
        'processing': {'type': 'fft', 'args': {
            'padding_duration': 900
        }},
        'signals': ["signals/eeg/eeg1"]
    },
    {
        'name': 'cycle_index_window',
        'processing': {'type': 'cycle_index_window', 'args': {
            'padding_duration': 900
        }},
        'signals': ["signals/eeg/eeg1"]
    },
    {
        'name': 'index_window',
        'processing': {'type': 'index_window', 'args': {
            'padding_duration': 900
        }},
        'signals': ["signals/eeg/eeg1"]
    }
]

features_description = {
    "fft": {
        "shape": [
            1876
        ]
    },
    "index_window": {
        "shape": [
            1
        ]
    },
    "cycle_index_window": {
        "shape": [
            5
        ]
    },

}

normalization = {"signals": [], 'features': [{
    'name': 'fft',
    'normalization': [{
        'type': 'standardization',
        'args': {}
    }
    ]
}, {
    'name': 'index_window',
    'normalization': [
    ]
}, {
    'name': 'cycle_index_window',
    'normalization': [
    ]
},

]}

models = [{'net_parameters': {
    'n_class': 5,
    'type': 'modulo_net',
    'encoders': {},
    'features_encoder': {'type': 'FullyConnected', 'args': {'layers': [10, 5]}},
    'reducer': {},
    'sequence_encoder': {'type': 'ResidualGRUSequenceEncoder', 'args': {'cells': 16}}
}, 'normalization': normalization
},
]

trainer_parameters = {
    "type": "base",

    'args': {
        'epochs': 10,
        'patience': 5,
        "optimizer": {"type": "adam", "args": {"lr": 1e-3}}
    }
}


def test_models_features():
    """
    Check that models are able to learn very simple patterns on randomly generated data.
    Chek that the models save and loads correctly
    """
    train_records, val_records = 5, 1
    temporal_context = 3

    # build dataset
    try:
        shutil.rmtree('/tmp/fake_memmmaps/')
    except:
        pass

    memmaps = generate_memmaps(train_records + val_records, three_groups_record_description,
                               features_memmap)

    for model_config in models:
        train_dataset = DreemDataset(groups_description={},
                                     features_description=features_description,
                                     records=memmaps[:train_records],
                                     temporal_context=temporal_context)
        val_dataset = DreemDataset(groups_description={}, features_description=features_description,
                                   records=memmaps[train_records:],
                                   temporal_context=temporal_context)

        print('Testing model', model_config['net_parameters'])
        normalization_parameters = model_config['normalization']
        normalization_parameters = initialize_standardization_parameters(
            train_dataset, normalization_parameters)

        net = ModuloNet(groups={}, features=features_description,
                        normalization_parameters=normalization_parameters,
                        net_parameters=model_config['net_parameters'])

        trainer = Trainer(net=net, **trainer_parameters['args'])
        # check that the model is learning
        metrics = trainer.train(train_dataset, val_dataset, verbose=0)
        metrics = trainer.validate(val_dataset)[0]

        # check the save and load of models work well
        net.save('/tmp/net.gz')
        net2 = ModuloNet.load('/tmp/net.gz')

        for p1, p2 in zip(net.parameters(), net2.parameters()):
            assert p1.data.ne(p2.data).sum() == 0

        # Check that the models predict hypnogram of the good shape with good perf
        preds = net2.predict_on_dataset(val_dataset)
        for record in val_dataset.records:
            print(preds)
            estimated = preds[record]
            real = val_dataset.hypnogram[record]
            assert len(estimated) == len(real)

        del net, net2, trainer, train_dataset, val_dataset
        gc.collect()
        torch.cuda.empty_cache()

        os.remove('/tmp/net.gz')
    shutil.rmtree('/tmp/fake_memmmaps/')
