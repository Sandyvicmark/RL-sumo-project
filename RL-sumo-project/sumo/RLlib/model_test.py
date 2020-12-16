import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy


class Actor(nn.Module):
    def __init__(self, n_inp, n_hidden):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_inp, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)

        self.mean = nn.Linear(n_hidden, 1)
        self.mean.weight.data.mul_(0.1)
        self.mean.bias.data.mul_(0.0)

        self.log_stddev = nn.Parameter(torch.zeros(1))

        self.module_list = [self.fc1, self.fc2, self.mean, self.log_stddev]
        self.module_list_old = [None] * 4

        # required so that start of episode does not throw error
        self.backup()

    def backup(self):
        for i in range(len(self.module_list)):
            self.module_list_old[i] = deepcopy(self.module_list[i])

    def forward(self, x, old=False):
        if not old:
            x = F.tanh(self.fc1(x))
            x = F.tanh(self.fc2(x))
            mu = self.mean(x)
            log_stddev = self.log_stddev.expand_as(mu)
            return mu.squeeze(), log_stddev.squeeze()
        else:
            x = F.tanh(self.module_list_old[0](x))
            x = F.tanh(self.module_list_old[1](x))
            mu = self.module_list_old[2](x)
            log_stddev = self.module_list_old[3].expand_as(mu)
            return mu.squeeze(), log_stddev.squeeze()


class Critic(nn.Module):
    def __init__(self, n_inp, n_hidden):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_inp, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)

        self.state_val = nn.Linear(n_hidden, 1)
        self.state_val.weight.data.mul_(0.1)
        self.state_val.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        state_val = self.state_val(x)
        return state_val
