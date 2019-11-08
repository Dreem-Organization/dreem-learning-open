from .gru_sequence_encoder import GRUSequenceEncoder, LSTMSequenceEncoder
from .residual_sequence_encoder import ResidualGRUSequenceEncoder, \
    DeepSleepNetResidualSequenceEncoder
from .temporal_flattener import TemporalFlattener

sequence_encoders = {
    'GRUSequenceEncoder': GRUSequenceEncoder,
    'LSTMSequenceEncoder': LSTMSequenceEncoder,
    'ResidualGRUSequenceEncoder': ResidualGRUSequenceEncoder,
    'TemporalFlattener': TemporalFlattener,
    'DeepSleepNetResidualSequenceEncoder': DeepSleepNetResidualSequenceEncoder
}
