import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.actor_optm = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optm = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = device
        self.actor_loss_list = []
        self.critic_loss_list = []

    def take_action(self, state):
        if type(state) is tuple:
            state = torch.tensor([state[0]], dtype=torch.float32)
        else:
            state = torch.tensor([state], dtype=torch.float32)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def max_q_value(self, state):
        if type(state) is tuple:
            state = state[0]
        state = torch.tensor([state], dtype=torch.float32)
        return self.critic(state).max().item()
    
    # 关注gymnasium中state新增的info，变为字典，当只需要state这一个array的时候需要注意过滤和统一性
    def update(self, transition_dict):
        #print(f"transition: {transition_dict}")
        # print("States shape:", len(transition_dict['states']))
        # print("First few elements:", transition_dict['states'])
        # print("Any empty?", any(len(s) == 0 for s in transition_dict['states']))

        if type(transition_dict['states']) is tuple:
            states = torch.tensor(transition_dict['states'][0], dtype=torch.float32).to(self.device)
        else:
            states = torch.tensor(transition_dict['states'], dtype=torch.float32).to(self.device)

        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).view(-1, 1).to(self.device)

        with torch.no_grad():
            td_target = rewards + self.gamma * self.critic(next_states) * (1-dones)  
        now_v = self.critic(states)
        advantage = (td_target - now_v).detach()

        critic_loss = F.mse_loss(now_v, td_target)
        self.critic_optm.zero_grad()
        critic_loss.backward()
        self.critic_optm.step()
        self.critic_loss_list.append(critic_loss.item())

        log_probs = torch.log(self.actor(states).gather(1, actions) + 1e-8)
        actor_loss = -torch.mean(advantage * log_probs)
        self.actor_optm.zero_grad()
        actor_loss.backward()
        self.actor_optm.step()
        self.actor_loss_list.append(actor_loss.item())
        

