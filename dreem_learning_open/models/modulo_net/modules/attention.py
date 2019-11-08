import torch
import torch.nn as nn

activation_functions = {
    'tanh': nn.Tanh
}


class Attention(nn.Module):
    """"
    Attention module similarly to Luong 2015
    """

    def __init__(self, input_dim, context_size=32, activation='tanh'):
        """

        input_dim (int): Dimensions of the input/ Number of features_data
        context_size (int): number of dim to use from the context
        """
        super(Attention, self).__init__()
        context_matrix = torch.nn.init.xavier_uniform_(torch.Tensor(context_size, input_dim))
        context_bias = torch.nn.init.xavier_uniform_(torch.Tensor(context_size, 1)).transpose(0, 1)
        context_vector = torch.nn.init.xavier_uniform_(torch.Tensor(context_size, 1))

        self.context_matrix = torch.nn.Parameter(data=context_matrix,
                                                 requires_grad=True).float()
        self.context_bias = torch.nn.Parameter(data=context_bias,
                                               requires_grad=True).float()
        self.context_vector = torch.nn.Parameter(data=context_vector,
                                                 requires_grad=True).float()

        self.softmax = torch.nn.Softmax(dim=1)
        self.tanh = activation_functions[activation]()
        print('Using ' + activation + ' as activation')
        print(self.tanh)

    def forward(self, x):
        """

        x (tensor: batch_size,sequence length,input_dim):
        returns x (tensor: batch_size,input_dim),  attention_weights (tensor: batch_size,sequence_length)
        """
        batch_size, length, n_features = x.shape

        x_att = x.reshape(-1, n_features)
        u = x_att.mm(self.context_matrix.transpose(0, 1)) + self.context_bias
        u = self.tanh(u)
        uv = u.mm(self.context_vector)
        uv = uv.view(batch_size, length)
        alpha = torch.nn.Softmax(dim=1)(uv)
        alpha = alpha.unsqueeze(-1)
        x_out = alpha * x
        x_out = x_out.sum(1)

        return x_out, alpha
