import torch.nn as nn
from torch import torch

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        out, hidden = self.rnn(input, hidden)
        out = self.fc(out.contiguous().view(out.size(0) * out.size(1), out.size(2)))
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)

class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        out, hidden = self.lstm(input, hidden)
        out = self.fc(out.contiguous().view(out.size(0) * out.size(1), out.size(2)))
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size),
                torch.zeros(self.n_layers, batch_size, self.hidden_size))
