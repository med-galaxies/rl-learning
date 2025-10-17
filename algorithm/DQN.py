import random
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        #print(f"形状 {x.shape}, 维度 {x.dim()} {x}")
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class VANet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VANet, self).__init__()
        self.shared_fc = nn.Linear(state_dim, hidden_dim)
        self.v_fc = nn.Linear(hidden_dim, action_dim)
        self.a_fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        A = self.a_fc(F.relu(self.shared_fc(x)))
        V = self.v_fc(F.relu(self.shared_fc(x)))
        Q = V + A - A.mean(1).view(-1, 1)
        return Q

class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device, type="dueling"):
        self.action_dim = action_dim
        if type == "dueling":
            self.q_net = VANet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
            # 目标网络
            self.target_q_net = VANet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        else:
            self.q_net = QNet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
            # 目标网络
            self.target_q_net = QNet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        self.loss_list = []

    def take_action(self, states):

        if type(states) is tuple:
            states = torch.tensor([states[0]], dtype=torch.float32)
        elif isinstance(states, np.ndarray):
            states = torch.tensor([states], dtype=torch.float32)

        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            action = self.q_net(states).argmax().item()
        return action
    
    def max_q_value(self, state):
        #print(f"state: {state}")
        if type(state) is tuple:
            state = state[0]
        state = torch.tensor([state], dtype=torch.float32)
        return self.q_net(state).max().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        
        with torch.no_grad():
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)
        
        loss = F.mse_loss(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_list.append(loss.item())
        
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
        
        return loss.item()
    