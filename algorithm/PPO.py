import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import rl_utils

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ValueNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.model(x)
    
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=1)
        ) 

    def forward(self, x):
        return self.model(x)
    
class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mean = F.tanh(self.fc_mean(x)) * 2.0
        log_std = F.softplus(self.fc_log_std(x)) + 1e-5
        std = torch.exp(torch.clamp(log_std, -5, 2))
        return mean, std
    
class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, device, action_type, \
                 gamma, lmbda, epoch, epsilon):
        self.action_type = action_type
        if action_type == 'Discrete':
            self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        else:
            self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.device = device
        self.gamma = gamma
        self.lmbda = lmbda
        self.epoch = epoch
        self.epilson = epsilon
        self.critic_loss_list = []
        self.actor_loss_list = []

    def is_continuous(self):
        return self.action_type == 'Continuous'
    
    def take_action(self, state):
        if isinstance(state, tuple):
            state = state[0]
        state = torch.tensor([state], dtype=torch.float32).to(self.device)
        if not self.is_continuous():
            probs = self.actor(state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            return action.item()
        else:
            mean, std = self.actor(state)
            action_dist = torch.distributions.Normal(mean, std)
            actions = action_dist.sample()
            return [actions.item()]
        
    def max_q_value(self, state):
        if type(state) is tuple:
            state = state[0]
        state = torch.tensor([state], dtype=torch.float32)
        return self.critic(state).max().item()
        
    def update(self, transition_dict):
        states = transition_dict['states']
        if isinstance(states, tuple):
            states = states[0]
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(transition_dict['next_states']).to(self.device)
        rewards = torch.FloatTensor(transition_dict['rewards']).view(-1, 1).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).view(-1, 1).to(self.device)
        dones = torch.FloatTensor(transition_dict['dones']).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states).detach() * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        if not self.is_continuous():
            dist = torch.distributions.Categorical(self.actor(states))
            old_log_probs = dist.log_prob(actions.squeeze()).unsqueeze(1).detach()
            # old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        else:
            mean, std = self.actor(states)
            dist = torch.distributions.Normal(mean, std)
            old_log_probs = dist.log_prob(actions).detach()

        for _ in range(int(self.epoch)):
            if not self.is_continuous():
                new_log_probs = torch.log(self.actor(states).gather(1, actions))
            else:
                mean, std = self.actor(states)
                # print(f"mean: {torch.isnan(mean).any()}, std: {torch.isnan(std).any()}")
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            cmp1 = ratio * advantage
            cmp2 = torch.clamp(ratio, 1-self.epilson, 1+self.epilson) * advantage
            actor_loss = -torch.mean(torch.min(cmp1, cmp2))
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_loss_list.append(actor_loss.item())
            # print(f"actor loss:{actor_loss.item()}")

            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_loss_list.append(critic_loss.item())
            self.critic_optimizer.step()
            # print(f"critic loss: {critic_loss.item()}")

            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)




