import gc
import os
import shutil

import torch
from sklearn.metrics import cohen_kappa_score

from .descriptions import three_groups_record_description, memmaps_description_nested, \
    groups_description
from .utils import generate_memmaps
from ..datasets.dataset import DreemDataset
from ..models.modulo_net import ModuloNet
from ..models.modulo_net.normalization import initialize_standardization_parameters
from ..trainers.trainer import Trainer

defaut_normalization_parameters = {"signals": [{
    'name': 'eeg-eog',
    'normalization':
        [
            {
                'type': 'clip_and_scale',
                'args': {'min_value': -2, 'max_value': 2}
            }
        ]
}], 'features': []}

spectrogram_normalization_parameters = {"signals": [{
    'name': 'eeg-eog',
    'normalization':
        [
            {
                'type': 'spectrogram',
                'args': {'fs': 100, 'logpower': True}
            },
            {
                'type': 'standardization',
                'args': {}
            },
        ]
}], 'features': []}

signal_normalization_parameters = {"signals": [{
    'name': 'eeg-eog',
    'normalization':
        [{
            'type': 'clip_and_scale',
            'args': {'min_value': -2, 'max_value': 2}
        },
            {
                'type': 'affine',
                'args': {'gain': 1.5}
        },
            {
                'type': 'standardization',
                'args': {}
        }
        ]
}], 'features': []}

models = [
    {'net_parameters': {
        'n_class': 5,
        'type': 'modulo_net',
        'encoders': {"eeg-eog":
                     {
                         'type': "SeqSleepEpochEncoder",
                         'args': {'hidden_layers': 8, 'filter_dim': 8, 'bidir': True}
                     }
                     },
        'reducer': {'type': 'FlattenReducer', 'args': {}},
        'sequence_encoder': {'type': 'DeepSleepNetResidualSequenceEncoder', 'args': {'cells': 16}}
    }, 'normalization': spectrogram_normalization_parameters},
    {'net_parameters': {
        'n_class': 5,
        'type': 'modulo_net',
        'encoders': {"eeg-eog":
                     {
                         'type': "SimpleSleepNetEpochEncoderWithoutFrequencyReduction",
                         'args': {'hidden_layers': 8, 'bidir': True}
                     }
                     },
        'reducer': {'type': 'FlattenReducer', 'args': {}},
        'sequence_encoder': {'type': 'LSTMSequenceEncoder', 'args': {'cells': 16}}
    }, 'normalization': spectrogram_normalization_parameters},
    {'net_parameters': {
        'n_class': 5,
        'type': 'modulo_net',
        'encoders': {"eeg-eog":
                     {
                         'type': "SimpleSleepNetEpochEncoderWithoutChannelRecombination",
                         'args': {'hidden_layers': 8, 'bidir': True}
                     }
                     },
        'reducer': {'type': 'FlattenReducer', 'args': {}},
        'sequence_encoder': {'type': 'LSTMSequenceEncoder', 'args': {'cells': 16}}
    }, 'normalization': spectrogram_normalization_parameters},

]

models += [
    {'net_parameters': {
        'n_class': 5,
        'type': 'modulo_net',
        'encoders': {"eeg-eog":
                     {
                         'type': "DeepSleepEpochEncoder",
                         'args': {'cells': 16}
                     }
                     },
        'reducer': {'type': 'AttentionReducer', 'args': {'context_size': 8}},
        'sequence_encoder': {'type': 'LSTMSequenceEncoder', 'args': {'cells': 16}}
    }, 'normalization': signal_normalization_parameters},
    {'net_parameters': {
        'n_class': 5,
        'type': 'modulo_net',
        'encoders': {"eeg-eog":
                     {
                         'type': "ChambonEpochEncoder",
                         'args': {}
                     }
                     },
        "reducer": {
            "type": "FlattenReducer",
            "args": {}
        },
        "sequence_encoder": {
            "type": "TemporalFlattener",
            "args": {}
        }
    }, 'normalization': signal_normalization_parameters},
    {'net_parameters': {
        'n_class': 5,
        'type': 'modulo_net',
        'encoders': {"eeg-eog":
                     {
                         'type': "TsinalisEpochEncoder",
                         'args': {}
                     }
                     },
        "reducer": {
            "type": "FlattenReducer",
            "args": {}
        },
        "sequence_encoder": {
            "type": "TemporalFlattener",
            "args": {}
        }
    }, 'normalization': signal_normalization_parameters}

]

models += [{'net_parameters': {
    'n_class': 5,
    'type': 'modulo_net',
    'encoders': {"eeg-eog":
                 {
                     'type': "SimpleSleepEpochEncoder",
                     'args': {'hidden_layers': 8, 'filter_dim': 8, 'bidir': True}
                 }
                 },
    'reducer': {'type': 'PoolReducer', 'args': {'pool_operation': 'max'}},
    'sequence_encoder': {'type': 'GRUSequenceEncoder', 'args': {'cells': 16}}
}, 'normalization': spectrogram_normalization_parameters},
]

trainer_parameters = {
    "type": "base",
    'args': {
        'epochs': 10,
        'patience': 10,
        "optimizer": {"type": "adam", "args": {"lr": 1e-3}}
    }
}


def test_models_default():
    """
    Check that models are able to learn very simple patterns on randomly generated data.
    Chek that the models save and loads correctly
    """
    groups = groups_description
    train_records, val_records = 5, 1
    temporal_context = 3

    # build dataset

    memmaps = generate_memmaps(train_records + val_records, three_groups_record_description,
                               memmaps_description_nested)

    for model_config in models:
        if model_config['net_parameters']['encoders']["eeg-eog"]['type'] in ["ChambonEpochEncoder",
                                                                             "TsinalisEpochEncoder"]:
            temporal_context_mode = 'concatenated'
            model_config['net_parameters']['input_temporal_context'] = temporal_context
        else:
            temporal_context_mode = 'sequential'
        train_dataset = DreemDataset(groups, features_description={},
                                     records=memmaps[:train_records],
                                     temporal_context=temporal_context,
                                     temporal_context_mode=temporal_context_mode)
        val_dataset = DreemDataset(groups, features_description={}, records=memmaps[train_records:],
                                   temporal_context=temporal_context,
                                   temporal_context_mode=temporal_context_mode)

        print('Testing model', model_config['net_parameters'])
        normalization_parameters = model_config['normalization']
        normalization_parameters = initialize_standardization_parameters(
            train_dataset, normalization_parameters)

        net = ModuloNet(train_dataset.groups_description, features={},
                        normalization_parameters=normalization_parameters,
                        net_parameters=model_config['net_parameters'])
        trainer = Trainer(net=net, **trainer_parameters['args'])

        trainer.reset_optimizer()
        # check that the model is learning
        metrics = trainer.train(train_dataset, val_dataset, verbose=1)
        metrics = net.predict_on_dataset(val_dataset, return_prob=True)
        metrics = trainer.validate(val_dataset)[0]

        assert metrics['accuracy'] >= 0.25
        assert metrics['cohen_kappa'] >= 0.25
        # check the save and load of models work well
        net.save('/tmp/net.gz')
        net2 = ModuloNet.load('/tmp/net.gz')

        for p1, p2 in zip(net.parameters(), net2.parameters()):
            assert p1.data.ne(p2.data).sum() == 0

        # Check that the models predict hypnogram of the good shape with good perf
        preds = net2.predict_on_dataset(val_dataset)
        for record in val_dataset.records:
            try:
                estimated = preds[record]
            except TypeError:
                estimated = preds[0][record]
            real = val_dataset.hypnogram[record]
            assert len(estimated) == len(real)
            assert cohen_kappa_score(estimated[real != -1], real[real != -1]) > 0.25

        del net, net2, trainer, train_dataset, val_dataset
        gc.collect()
        torch.cuda.empty_cache()

        os.remove('/tmp/net.gz')
    shutil.rmtree('/tmp/fake_memmmaps/')
