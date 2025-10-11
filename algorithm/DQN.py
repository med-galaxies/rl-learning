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
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        if type(state) is tuple:
            state = state[0]
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        #print(f"buffer state sampling:{state}")
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)
    

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
    
    def update(self, transition_dict):
        if type(transition_dict['states']) is tuple:
            states = torch.from_numpy(transition_dict['states'][0]).to(self.device)
        elif isinstance(transition_dict['states'], np.ndarray):
            states = torch.from_numpy(transition_dict['states']).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(1, -1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(1,-1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states']).to(self.device)
        dones = torch.tensor(transition_dict['dones']).view(1,-1).to(self.device)

        max_target_q = self.target_q_net(next_states).max(1)[0].view(1,-1).to(self.device)
        whole_target_q = rewards + self.gamma * max_target_q * (~dones).float()   
        q = self.q_net(states).view(1, -1).gather(1, actions).to(self.device)
        loss = torch.mean(F.mse_loss(whole_target_q, q))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1