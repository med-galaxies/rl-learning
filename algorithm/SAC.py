import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class PolicyNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        std = torch.clamp(std, min=1e-1, max=1)
        dist = Normal(mu, std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = dist.log_prob(normal_sample)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        # log_prob = torch.sum(dist.log_prob(normal_sample), dim=-1, keepdim=True)
        action = action * self.action_bound
        return action, log_prob

class QNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNetContinuous, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)

class SACContinuous:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, 
                 alpha_lr, target_entropy, tau, gamma, device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic1 = QNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic2 = QNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic1 = QNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic2 = QNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float, device=device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.loss_list1 = []
        self.loss_list2 = []
        self.loss_list_alpha = []

    def max_q_value(self, state, new_actions):
        if type(state) is tuple:
            state = state[0]
        state = torch.tensor([state], dtype=torch.float32).to(self.device)
        new_actions = torch.tensor(new_actions, dtype=torch.float).view(-1, 1).to(self.device)
        # new_actions, _ = self.actor(state)
        return min(self.critic1(state, new_actions).max().item(), self.critic2(state, new_actions).max().item())

    def take_action(self, x):
        if isinstance(x, tuple):
            x = x[0]
        state = torch.tensor([x], dtype=torch.float).to(self.device)
        action = self.actor(state)[0]
        return action.detach().cpu().numpy().flatten()
        # return [action.item()]
    
    def cal_target(self, rewards, next_states, dones):
        next_actions, next_log_pi = self.actor(next_states)
        entropy = -next_log_pi
        q1_value = self.target_critic1(next_states, next_actions)
        q2_value = self.target_critic2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
            
    def update(self, transition_dict):
        states = transition_dict['states']
        if isinstance(states, tuple):
            states = states[0]
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(transition_dict['next_states']).to(self.device)
        rewards = torch.FloatTensor(transition_dict['rewards']).view(-1, 1).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.FloatTensor(transition_dict['dones']).view(-1, 1).to(self.device)
        rewards = (rewards ) / 10.0

        td_target = self.cal_target(rewards, next_states, dones)

        # 计算 TD error (用于 PER)
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        td_error = (torch.min(current_q1, current_q2) - td_target.detach()).cpu().detach().numpy()

        critic_1_loss = torch.mean(F.mse_loss(self.critic1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(self.critic2(states, actions), td_target.detach()))
        self.loss_list1.append(critic_1_loss.item())
        self.loss_list2.append(critic_2_loss.item())
        self.critic1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic2_optimizer.step()

        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic1(states, new_actions)
        q2_value = self.critic2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)
        self.loss_list_alpha.append(alpha_loss.item())

        return None, td_error
        