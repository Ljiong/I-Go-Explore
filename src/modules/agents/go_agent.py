import torch.nn as nn
import torch.nn.functional as F


class GoExploreAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GoExploreAgent, self).__init__()
        self.args = args

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h

