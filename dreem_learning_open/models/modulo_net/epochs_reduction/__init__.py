from .attention_reducer import AttentionReducer
from .pool_reducer import PoolReducer, FlattenReducer

reducers = {
    'AttentionReducer': AttentionReducer,
    'PoolReducer': PoolReducer,
    'FlattenReducer': FlattenReducer
}
