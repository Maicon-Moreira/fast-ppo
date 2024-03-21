import torch as t
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.linear1 = nn.Linear(state_dim, 256)
        self.actor_linear = nn.Linear(256, action_dim)
        self.critic_linear = nn.Linear(256, 1)

        for layer in [self.linear1, self.actor_linear, self.critic_linear]:
            t.nn.init.orthogonal_(layer.weight, t.nn.init.calculate_gain("relu"))
            t.nn.init.constant_(layer.bias, 0)

    def actor(self, state):
        x = F.relu(self.linear1(state))
        return F.softmax(self.actor_linear(x), dim=-1)

    def critic(self, state):
        x = F.relu(self.linear1(state))
        return self.critic_linear(x)
