from torch import nn


class GRUWithSkipConnection(nn.Module):

    def __init__(self, input_size, cells, dropout=0., bidir=True):
        super().__init__()
        self.hidden_size = cells
        self.residual_connection = nn.Linear(input_size, cells * (1 + bidir))
        self.gru = nn.GRU(input_size, cells, bidirectional=bidir)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden_state=None):
        x_residual = self.residual_connection(x)
        x = x.transpose(0, 1)
        x_lstm, hidden_state = self.gru(x, hidden_state)
        x_lstm = x_lstm.transpose(0, 1)
        x_cat = (x_residual + x_lstm) / 2
        x_cat = self.dropout(x_cat)
        return x_cat, hidden_state


class ResidualGRUSequenceEncoder(nn.Module):

    def __init__(self, feature_map_size, dropout=0.5, bidir=True, layers=2, cells=64):
        super().__init__()

        self.hidden_size = cells
        self.layers = layers
        self.output_size = cells * (1 + bidir)
        self.encoder = nn.ModuleList(
            [

                GRUWithSkipConnection(feature_map_size if k == 0 else cells * (1 + bidir)
                                      , cells, dropout, bidir=bidir) for k in range(layers)
            ]
        )

    def forward(self, x, hidden_state=None):
        if hidden_state is None:
            hidden_state = [hidden_state] * self.layers

        for i, layer in enumerate(self.encoder):
            x, hidden_state[i] = layer(x, hidden_state[i])
        return x, hidden_state


class DeepSleepNetResidualSequenceEncoder(nn.Module):

    def __init__(self, input_size, cells, dropout=0., layers=1, bidir=True):
        super().__init__()
        self.hidden_size = cells
        self.residual_connection = nn.Linear(input_size, cells * (1 + bidir))
        self.num_layers = layers
        self.lstm = nn.LSTM(input_size, cells, bidirectional=bidir, num_layers=layers,
                            dropout=dropout)

        self.hidden_size = cells
        self.output_size = cells * (1 + bidir)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden_state=None):
        x = self.dropout(x)
        x_residual = self.residual_connection(x)
        x = x.transpose(0, 1)
        x_lstm, hidden_state = self.lstm(x, hidden_state)
        x_lstm = x_lstm.transpose(0, 1)
        x_lstm = self.dropout(x_lstm)
        x_cat = (x_residual + x_lstm)
        x_cat = self.dropout(x_cat)
        return x_cat, hidden_state
