from .ablation.simple_net_without_channel_recombination import \
    SimpleSleepNetEpochEncoderWithoutChannelRecombination
from .ablation.simple_net_without_frequency_reduction import \
    SimpleSleepNetEpochEncoderWithoutFrequencyReduction
from .ablation.simple_sleep_net_with_filter_bank import \
    SimpleSleepNetWithFilterBank
from .legacy.chambon_net import ChambonEpochEncoder
from .legacy.deep_sleep_net_encoder import DeepSleepNetEpochEncoder
from .legacy.seq_sleep_net import SeqSleepNetEpochEncoder
from .legacy.tsinalis_net import TsinalisEpochEncoder
from .simple_sleep_net import SimpleSleepNetEpochEncoder

epoch_encoders = {
    "SeqSleepEpochEncoder": SeqSleepNetEpochEncoder,
    "DeepSleepEpochEncoder": DeepSleepNetEpochEncoder,
    "SimpleSleepEpochEncoder": SimpleSleepNetEpochEncoder,
    "ChambonEpochEncoder": ChambonEpochEncoder,
    'TsinalisEpochEncoder': TsinalisEpochEncoder,
    'SimpleSleepNetEpochEncoderWithoutChannelRecombination': SimpleSleepNetEpochEncoderWithoutChannelRecombination,
    'SimpleSleepNetEpochEncoderWithoutFrequencyReduction':
        SimpleSleepNetEpochEncoderWithoutFrequencyReduction,
    'SimpleSleepNetEpochEncoderWithFilterBank': SimpleSleepNetWithFilterBank
}
