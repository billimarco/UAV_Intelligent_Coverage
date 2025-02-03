import torch
import torch.nn as nn
import torch.nn.functional as F

hidden_sizes = [256, 256]


class DeepQNet(nn.Module):
    def __init__(self, state_dim=16, action_dim=2):
        super(DeepQNet, self).__init__()
        self.fl1 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
        self.fl2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fl3 = nn.Linear(hidden_sizes[1], 1)  # Output Q(s, a)

    def forward(self, state, action):
        combined = torch.cat([state, action], dim=1)
        # Process state through its branch
        out = F.relu(self.fl1(combined))
        out = F.relu(self.fl2(out))
        # return Q-value(s, a)
        return self.fl3(out)


class DoubleDeepQNet(nn.Module):
    def __init__(self, state_dim=16, action_dim=2):
        super(DoubleDeepQNet, self).__init__()
        self.fl1Q1 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
        self.fl2Q1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fl3Q1 = nn.Linear(hidden_sizes[1], 1)  # Output Q1(s, a)

        self.fl1Q2 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
        self.fl2Q2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fl3Q2 = nn.Linear(hidden_sizes[1], 1)  # Output Q2(s, a)

    def forward(self, state, action):
        combined = torch.cat([state, action], dim=1)

        outQ1 = F.relu(self.fl1Q1(combined))
        outQ1 = F.relu(self.fl2Q1(outQ1))

        outQ2 = F.relu(self.fl1Q2(combined))
        outQ2 = F.relu(self.fl2Q2(outQ2))

        # return Q-value(s, a) of first and second Q net
        return self.fl3Q1(outQ1), self.fl3Q2(outQ2)
