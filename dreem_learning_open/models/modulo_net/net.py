import copy
import json
import tarfile
import tempfile

import numpy as np
import torch
import tqdm
from torch import nn

from .epochs_encoder import epoch_encoders
from .epochs_reduction import reducers
from .features_encoder import features_encoders
from .normalization import normalize, normalize_features
from .sequence_encoder import sequence_encoders
from ...datasets.dataset import DreemDataset


class ModuloNet(nn.Module):
    """ Class to follow in order to use below train function

    Essentially: init and forward ok
    - get_args: to retrieve args for forward function from a batch of data (out of dataloader)
    """

    def __init__(self, groups, features, normalization_parameters, net_parameters):
        """

        groups: (dict) Description of the groups (similar to Dataset.group_description)
        normalization_parameters: (DatasetNormalizationPipeline) ust be compatible with group descriptions
        net_parameters: (dict) parameters to be feed to modify the architecture of the net
        """

        super(ModuloNet, self).__init__()
        self.groups = groups
        self.features = features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.methods = {}

        if isinstance(normalization_parameters['signals'], list) and isinstance(
                normalization_parameters['features'],
                list):
            self.normalization_parameters = normalization_parameters
        else:
            raise ValueError('normalization_parameters should be a list')

        self.net_parameters = {'n_class': 5, 'output_mode': 'one', 'input_temporal_context': 1}

        for param in net_parameters:
            self.net_parameters[param] = net_parameters[param]

        self.input_temporal_context = self.net_parameters['input_temporal_context']
        assert self.groups.keys() == self.net_parameters['encoders'].keys()

        for i, group in enumerate(self.normalization_parameters['signals']):
            for operation in group['normalization']:
                for arg_name, value in operation['args'].items():
                    if isinstance(value, torch.Tensor):
                        operation['args'][arg_name] = value.to(self.device)

        for i, feature in enumerate(self.normalization_parameters['features']):
            for operation in feature['normalization']:
                for arg_name, value in operation['args'].items():
                    if isinstance(value, torch.Tensor):
                        operation['args'][arg_name] = value.to(self.device)

        self.groups_encoder = {}

        self.output_mode = self.net_parameters['output_mode']
        if 'eval_output_mode' in self.net_parameters:
            self.eval_output_mode = self.net_parameters['eval_output_mode']
        else:
            self.eval_output_mode = self.net_parameters['output_mode']

        self.init_net()
        self.to(self.device)

        print('Output mode is ', self.output_mode)
        print('Eval Output mode is ', self.eval_output_mode)

    def init_net(self):

        encoders_input_shape = {}
        feature_map_size = 0
        if len(self.groups) > 0:
            for group, value in self.groups.items():
                value = copy.deepcopy(value)
                value['shape'][0] *= self.net_parameters['input_temporal_context']
                encoders_input_shape[group] = np.random.normal(
                    size=(1, 1) + tuple(value['shape'][::-1]))
                encoders_input_shape[group] = torch.from_numpy(
                    encoders_input_shape[group]).float().to(
                    self.device)
            encoders_input_shape = normalize(encoders_input_shape,
                                             self.normalization_parameters['signals'])
            for group in self.groups:
                self.groups[group]['encoder_input_shape'] = encoders_input_shape[group].size()

            for group in self.groups:
                encoder_type = self.net_parameters['encoders'][group]['type']
                encoder_args = self.net_parameters['encoders'][group]['args']
                self.groups_encoder[group] = epoch_encoders[encoder_type](self.groups[group],
                                                                          encoder_args)
                encoders_input_shape[group] = self.groups_encoder[group](
                    encoders_input_shape[group])
                self.groups[group]['reducer_input_shape'] = encoders_input_shape[group].size()[1:]

            for key, module in self.groups_encoder.items():  # register blocks
                self.add_module(key, module)

            reducer_type = self.net_parameters['reducer']['type']
            reducer_args = self.net_parameters['reducer']['args']
            self.reducer = reducers[reducer_type](self.groups, **reducer_args)
            self.reducer.to(self.device)
            feature_map_size += self.reducer(encoders_input_shape).size()[-1]
        else:
            self.groups_encoder = None

        if len(self.features) > 0:
            assert 'features_encoder' in self.net_parameters, 'Features are specified, but there are no features encoder'
            feature_encoder_type = self.net_parameters['features_encoder']['type']
            feature_encoder_args = self.net_parameters['features_encoder']['args']
            self.features_encoder = features_encoders[feature_encoder_type](self.features,
                                                                            **feature_encoder_args)
        else:
            self.features_encoder = None

        if self.features_encoder is not None:
            feature_map_size += self.features_encoder.out_features

        sequence_encoder_cls = sequence_encoders[self.net_parameters['sequence_encoder']['type']]
        sequence_encoder_params = self.net_parameters['sequence_encoder']['args']
        self.sequence_encoder = sequence_encoder_cls(feature_map_size, **sequence_encoder_params)
        self.classifier = nn.Linear(self.sequence_encoder.output_size,
                                    self.net_parameters['n_class'])

    def forward_features(self, x):
        """
        Reshape and forward the features_data from forward features_data into a sequence of feature
        x:
        """
        features = {}
        for group, encoder in self.groups_encoder.items():
            batch_size, temporal_context = x[group].size()[:2]
            features[group] = encoder(x[group])
        features = self.reducer(features)
        features = features.view((batch_size, temporal_context, -1))
        return features

    def forward(self, x, sequence_hidden_state=None):

        if self.training:
            output_mode = self.output_mode
        else:
            output_mode = self.eval_output_mode

        features = []
        if self.groups_encoder is not None:
            signals = normalize(x['signals'], self.normalization_parameters['signals'])
            features += [self.forward_features(signals)]

        if self.features_encoder is not None:
            epoch_features = normalize_features(x['features'],
                                                self.normalization_parameters['features'])
            epoch_features = self.features_encoder.forward(epoch_features)
            features += [epoch_features]

        features = torch.cat(features, -1)

        seq, sequence_hidden_state = self.sequence_encoder.forward(features, sequence_hidden_state)
        out = self.classifier(seq)
        if output_mode == 'many':
            out = out.view(-1, self.net_parameters['n_class'])
        elif output_mode == 'one':
            if len(out.shape) == 2:
                # No need to select the right step when there is no temporal dimension in the output
                pass
            elif len(out.shape) == 3:
                # Select the middle of the sequence
                batch_size, temporal_context, n_class = out.shape
                out = out[:, temporal_context // 2, :]

        return out, sequence_hidden_state

    def get_args(self, data):
        """Input batch of data, output args for forward method and loss.

        data = next(iter(dataloader))
        args, hypnogram = net.get_args(data)
        output = net.forward(args)
        loss(output, hypnogram)
        """

        if self.training:
            output_mode = self.output_mode
        else:
            output_mode = self.eval_output_mode
        assert output_mode in ['many', 'one'], 'Mode has to be either many, either one'

        args = {'signals': {}, 'features': {}}

        args['signals'] = {group: data['groups'][group].to(self.device, non_blocking=True) for group
                           in self.groups}
        if self.features_encoder is not None:
            args['features'] = {
                feature: data['features'][feature].to(self.device, non_blocking=True) for feature in
                self.features}
        args = (args,)

        if output_mode == 'one':
            for group in self.groups:
                temporal_context = data['groups'][group].shape[1]
            for signal in self.features:
                temporal_context = data['features'][signal].shape[1]
            central_epoch = temporal_context // 2
            hypnogram = data['hypnogram'][:, central_epoch].contiguous()
        elif output_mode == 'many':
            hypnogram = data['hypnogram'].view(-1)
        else:
            raise ValueError('Invalid output mode')

        hypnogram = hypnogram.to(self.device, non_blocking=True)

        return args, hypnogram

    def save(self, filename):
        """

        filename: (str) Save the net to filename under .tar format
            """
        with tarfile.open(filename, "w") as tar:
            # Net parameters
            name = "{}/net_params.json".format(tempfile.mkdtemp())
            json.dump(self.net_parameters, open(name, "w"))
            tar.add(name, arcname="net_params.json")

            # state
            name = "{}/state.torch".format(tempfile.mkdtemp())
            torch.save(self.state_dict(), name)
            tar.add(name, arcname="state.torch")

            # normalization pipeline
            name = "{}/normalization_parameters.json".format(tempfile.mkdtemp())
            normalization_parameters_save = copy.deepcopy(self.normalization_parameters)
            for var_type in ['signals', 'features']:
                for i, group in enumerate(normalization_parameters_save[var_type]):
                    for operation in group['normalization']:
                        for arg_name, value in operation['args'].items():
                            if isinstance(value, torch.Tensor):
                                operation['args'][arg_name] = value.cpu().numpy().tolist()

            json.dump(normalization_parameters_save, open(name, "w"))
            tar.add(name, arcname="normalization_parameters.json")

            # groups parameters
            name = "{}/groups.json".format(tempfile.mkdtemp())
            json.dump(self.groups, open(name, "w"))
            tar.add(name, arcname="groups.json")

            # feature parameters
            name = "{}/features.json".format(tempfile.mkdtemp())
            json.dump(self.features, open(name, "w"))
            tar.add(name, arcname="features.json")

        return filename

    def predict_on_record(self, record: str, dataset: DreemDataset, stride: int = 1,
                          return_prob: bool = False,

                          mode: str = 'arithmetic', return_hidden_state=False):
        self.eval()

        if self.training:
            output_mode = self.output_mode
        else:
            output_mode = self.eval_output_mode

        idx_min, idx_max = dataset.record_index_eval[record]

        record_length = idx_max - idx_min
        n_class = self.net_parameters['n_class']
        temporal_context = dataset.temporal_context
        temporal_offset_due_to_input_temporal_context = self.input_temporal_context // 2 * 2
        predicted_output = np.zeros(shape=(
            record_length + temporal_context + temporal_offset_due_to_input_temporal_context,
            n_class))
        record_hidden_states = []

        for batch, idxs in dataset.get_record(record, mode='eval', stride=stride,
                                              return_index=True):
            X = self.get_args(batch)[0]
            batch_size = len(idxs)
            preds = self.forward(*X)
            try:
                preds, hidden_state = preds[0].cpu().detach().numpy(), [
                    pred.cpu().detach().numpy() for pred in
                    preds[1]]
            except AttributeError:
                preds, hidden_state = preds[0].cpu().detach().numpy(), [
                    h[0].cpu().detach().numpy() for h in
                    preds[1]]
            record_hidden_states += [hidden_state]

            preds = preds.reshape((batch_size, -1, n_class))
            if mode == 'geometric':
                preds = np.log(np.exp(preds) / np.sum(np.exp(preds), axis=-1, keepdims=True))
            elif mode == 'arithmetic':
                preds = np.exp(preds) / np.sum(np.exp(preds), axis=-1, keepdims=True)
            elif mode == 'vote':
                preds = (preds == preds.max(axis=-1, keepdims=True)).astype(int)

            for i in range(batch_size):
                if output_mode == 'many':
                    predicted_output[idxs[i] - temporal_context // 2:
                                     idxs[i] + temporal_context // 2 + 1] += preds[i]

                elif output_mode == 'one':
                    predicted_output[idxs[i]:idxs[i] + 1] += preds[i]
                else:
                    raise ValueError('Invalid output mode')

        if return_prob:
            if return_hidden_state:
                return np.exp(predicted_output) / np.sum(np.exp(predicted_output), -1,
                                                         keepdims=True), record_hidden_states
            else:
                return np.exp(predicted_output) / np.sum(np.exp(predicted_output), -1,
                                                         keepdims=True)
        else:
            if return_hidden_state:
                return np.argmax(predicted_output, -1), record_hidden_states
            else:
                return np.argmax(predicted_output, -1)

    def predict_on_dataset(self, dataset: DreemDataset, stride: int = 1, return_prob: bool = False,
                           mode: str = 'geometric', verbose=False):
        self.eval()
        result = {}
        enumerator = dataset.records
        if verbose:
            enumerator = tqdm.tqdm(enumerator)
        for record in enumerator:
            result[record] = self.predict_on_record(record, dataset, stride, return_prob, mode)
        return result

    @classmethod
    def load(cls, filename):
        """

        filename: (str) load a net from filename
            """
        with tarfile.open(filename, "r") as tar:
            # Load net parameters
            net_parameters = json.loads(
                tar.extractfile("net_params.json").read().decode("utf-8"))

            # load normalization pipeline
            normalization_parameters = json.loads(
                tar.extractfile("normalization_parameters.json").read().decode("utf-8"))

            for var_type in ['signals', 'features']:
                for i, group in enumerate(normalization_parameters[var_type]):
                    for operation in group['normalization']:
                        if operation['type'] == 'standardize' or operation[
                            'type'] == 'standardization':
                            operation['args']['mu'] = torch.Tensor(operation['args']['mu'])
                            operation['args']['sigma'] = torch.Tensor(operation['args']['sigma'])

            groups = json.loads(
                tar.extractfile("groups.json").read().decode("utf-8"))

            features = json.loads(
                tar.extractfile("features.json").read().decode("utf-8"))

            # State
            path = tempfile.mkdtemp()
            tar.extract("state.torch", path=path)
            net = cls(groups, features, normalization_parameters, net_parameters)
            if torch.cuda.is_available():
                net.load_state_dict(torch.load(path + "/state.torch"))
            else:
                net = net.to('cpu')
                net.load_state_dict(torch.load(path + "/state.torch", map_location='cpu'))

        return net
