import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, state):
        return self.model(state)
    
class Reinforce:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.device = device
        self.loss_list = []

    def take_action(self, states):
        if type(states) is tuple:
                states = torch.tensor([states[0]], dtype=torch.float32)
        elif isinstance(states, np.ndarray):
            states = torch.tensor([states], dtype=torch.float32)
        probs = self.policy_net(states)
        action_dist = torch.distributions.Categorical(probs)
        actions = action_dist.sample()
        return actions.item()
    
    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        self.optimizer.zero_grad()
        G = 0
        for i in reversed(range(len(reward_list))):
            if type(state_list[i]) is tuple:
                state = torch.tensor([state_list[i][0]], dtype=torch.float32).to(self.device)
            else:
                state = torch.tensor([state_list[i]], dtype=torch.float32).to(self.device)
            reward = torch.tensor(reward_list[i], dtype=torch.float32).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)

            G = self.gamma * G + reward
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            loss = -log_prob * G
            loss.backward()
            self.loss_list.append(loss.item())
        self.optimizer.step()


