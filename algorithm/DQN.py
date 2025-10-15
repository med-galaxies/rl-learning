import random
import numpy as np
import collections
import torch
import torch.nn.functional as F

class QNet(torch.nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        #print(f"形状 {x.shape}, 维度 {x.dim()} {x}")
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
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

    def take_action(self, state):

        if type(state) is tuple:
            state = torch.from_numpy(state[0])
        elif isinstance(state, np.ndarray):
            state = torch.from_numpy(state)

        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            action = self.q_net(state).argmax().item()
        return action
    
    def max_q_value(self, state):
        #print(f"state: {state}")
        if type(state) is tuple:
            state = state[0]
        state = torch.tensor(state, dtype=torch.float32)
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
    