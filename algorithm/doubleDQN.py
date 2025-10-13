import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNet(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.model(x)
    
class DoubleDQN:

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, update_freq, device):
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_freq = update_freq
        self.device = device
        self.q_net = QNet(state_dim, hidden_dim, action_dim).to(device)
        self.q_target_net = QNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.count = 0
        self.loss_list = []

    def take_action(self, states):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            states = torch.tensor(states, dtype=torch.float32).to(self.device)
            action = self.q_net(states).argmax().item()
        return action
    
    def update(self, transition_dict):
        print(f"dict: {transition_dict}")
        states = torch.tensor(transition_dict["states"], dtype=torch.float32).to(self.device)
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict["next_states"], dtype=torch.float32).to(self.device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float32).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        max_action_idx = self.q_net(next_states).max(1)[1].view(-1, 1)
        max_next_q_values = self.q_target_net(next_states).gather(1, max_action_idx)
        td_target = rewards + self.gamma * max_next_q_values * (1-dones)
        
        self.optimizer.zero_grad()
        loss = F.mse_loss(q_values, td_target)
        loss.backward()
        self.optimizer.step()
        self.loss_list.append(loss.item())

        if self.count % self.update_freq == 0:
            self.q_target_net.load_state_dict(self.q_net.state_dict())
        
        self.count += 1
