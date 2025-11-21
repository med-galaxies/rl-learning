import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NoisyLinear(nn.Module):

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('weight_mu', torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_sigma', torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_mu', torch.FloatTensor(out_features))
        self.register_buffer('bias_sigma', torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.weight_sigma.size(1)))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class QNet(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim, isNoise=False):
        super(QNet, self).__init__()
        if not isNoise:
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, action_dim)
        else:
            self.fc1 = NoisyLinear(state_dim, hidden_dim)
            self.fc2 = NoisyLinear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
    
class VANet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, isNoise=False):
        super(VANet, self).__init__()
        if not isNoise:
            self.shared_fc = nn.Linear(state_dim, hidden_dim)
            self.v_fc = nn.Linear(hidden_dim, action_dim)
            self.a_fc = nn.Linear(hidden_dim, 1)
        else:
            self.shared_fc = NoisyLinear(state_dim, hidden_dim)
            self.v_fc = NoisyLinear(hidden_dim, action_dim)
            self.a_fc = NoisyLinear(hidden_dim, 1)
    
    def forward(self, x):
        A = self.a_fc(F.relu(self.shared_fc(x)))
        V = self.v_fc(F.relu(self.shared_fc(x)))
        Q = V + A - A.mean(1).view(-1, 1)
        return Q
    
    def reset_noise(self):
        self.shared_fc.reset_noise()
        self.v_fc.reset_noise()
        self.a_fc.reset_noise()
    
class DoubleNoiseDQN:

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, update_freq, device, type="dueling", buffer_type='per', isNoise=False):
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_freq = update_freq
        self.device = device
        if type == "dueling":
            self.q_net = VANet(state_dim, hidden_dim, action_dim, isNoise).to(device)
            self.q_target_net = VANet(state_dim, hidden_dim, action_dim, isNoise).to(device)
        else:
            self.q_net = QNet(state_dim, hidden_dim, action_dim, isNoise).to(device)
            self.q_target_net = QNet(state_dim, hidden_dim, action_dim, isNoise).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.count = 0
        self.loss_list = []
        self.buffer_type = buffer_type
        self.isNoise = isNoise

    def take_action(self, states):
        if not self.isNoise:
            if np.random.random() < self.epsilon:
                action = np.random.randint(self.action_dim)
            else:
                if type(states) is tuple:
                    states = torch.tensor([states[0]], dtype=torch.float32)
                elif isinstance(states, np.ndarray):
                    states = torch.tensor([states], dtype=torch.float32)
                action = self.q_net(states).argmax().item()
        else:
            if type(states) is tuple:
                states = torch.tensor([states[0]], dtype=torch.float32)
            elif isinstance(states, np.ndarray):
                states = torch.tensor([states], dtype=torch.float32)
            action = self.q_net(states).argmax().item()
        return action
    
    def max_q_value(self, state):
        if type(state) is tuple:
            state = state[0]
        state = torch.tensor([state], dtype=torch.float32)
        return self.q_net(state).max().item()
    
    def update(self, transition_dict):
        #print(f"dict: {transition_dict}")
        states = torch.tensor(transition_dict["states"], dtype=torch.float32).to(self.device)
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict["next_states"], dtype=torch.float32).to(self.device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float32).view(-1, 1).to(self.device)
        if self.buffer_type == 'per':
            weights = torch.tensor(transition_dict['weights'], dtype=torch.float32).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        max_action_idx = self.q_net(next_states).max(1)[1].view(-1, 1)
        max_next_q_values = self.q_target_net(next_states).gather(1, max_action_idx)
        td_target = rewards + self.gamma * max_next_q_values * (1-dones)
        
        self.optimizer.zero_grad()
        if self.buffer_type == 'per':
            loss = (q_values - td_target).pow(2) * weights
            loss = loss.mean()
        else:
            loss = F.mse_loss(q_values, td_target)
        loss.backward()
        self.optimizer.step()
        self.loss_list.append(loss.item())

        if self.isNoise:
            self.q_net.reset_noise()
            self.q_target_net.reset_noise()

        if self.count % self.update_freq == 0:
            self.q_target_net.load_state_dict(self.q_net.state_dict())
        
        self.count += 1
        return loss.item(), (td_target-q_values).detach().numpy()
