from torch import nn


class GRUSequenceEncoder(nn.Module):

    def __init__(self, input_size, cells=64, bidir=False, dropout=0.5, layers=1):
        super(GRUSequenceEncoder, self).__init__()
        self.sequence_encoder = nn.GRU(input_size=input_size,
                                       hidden_size=cells,
                                       bidirectional=bidir, batch_first=True, num_layers=layers,
                                       dropout=dropout)
        self.output_size = cells * (
                1 + bidir)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden_state=None):
        x, hidden_state = self.sequence_encoder(x, hidden_state)
        x = self.dropout(x)
        return x, hidden_state


class LSTMSequenceEncoder(nn.Module):

    def __init__(self, input_size, cells=64, bidir=False, dropout=0.5, layers=1):
        super(LSTMSequenceEncoder, self).__init__()
        self.sequence_encoder = nn.LSTM(input_size=input_size,
                                        hidden_size=cells,
                                        bidirectional=bidir, batch_first=True, num_layers=layers,
                                        dropout=dropout)
        self.output_size = cells * (
                1 + bidir)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden_state=None):
        x, hidden_state = self.sequence_encoder(x, hidden_state)
        x = self.dropout(x)
        return x, hidden_state
