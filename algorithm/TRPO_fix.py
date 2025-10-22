import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils import rl_utils

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
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

class TRPO:
    def __init__(self, state_dim, hidden_dim, action_dim, lmbda, kl_constriant, alpha,
                 critic_lr, gamma, device):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.lmbda = lmbda
        self.kl_constriant = kl_constriant
        self.alpha = alpha
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.device = device
        self.critic_loss_list = []
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def take_action(self, state):
        if isinstance(state, tuple):
            state = state[0]
        state = torch.tensor([state], dtype=torch.float32).to(self.device)
        probs = self.actor(state)
        # print("state.shape:", state.shape)
        # print("probs.shape:", probs.shape)
        # print("probs.sum(dim=1):", probs.sum(dim=1))
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        if isinstance(states, tuple):
            states = states[0]
        new_action_dist = torch.distributions.Categorical(self.actor(states))
        kl = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists, new_action_dist)
        )
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])

        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector
    
    def conjugate_gradient(self, grad, states, old_action_dists):
        if isinstance(states, tuple):
            states = states[0]
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r,r)
        for i in range(10):
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x
    
    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):  # 计算策略目标
        if isinstance(states, tuple):
            states = states[0]
        log_probs = torch.log(actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio*advantage)
    
    def line_search(self, states, actions, advantage, old_log_probs, old_action_dists, max_vec):  # 线性搜索
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage,
                                     old_log_probs, self.actor)
        for i in range(15):
            coef = self.alpha ** i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(new_para, new_actor.parameters())
            new_action_dists = torch.distributions.Categorical(new_actor(states))
            kl_div = torch.mean(
                torch.distributions.kl.kl_divergence(old_action_dists,new_action_dists)
            )
            new_obj = self.compute_surrogate_obj(states, actions, advantage,
                                         old_log_probs, new_actor)
            if new_obj > old_obj and kl_div < self.kl_constriant:
                return new_para
        return old_para
    
    def policy_learn(self, states, actions, old_action_dists, old_log_probs, advantage):  # 更新策略函数
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor) #计算梯度g
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters()) #计算梯度g
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach() #计算梯度g
        descent_direction = self.conjugate_gradient(obj_grad, states, old_action_dists) #计算H^-1 * g
        Hd = self.hessian_matrix_vector_product(states, old_action_dists, descent_direction) #计算g^T * H^-1 * g
        max_coef = torch.sqrt(self.kl_constriant / (torch.dot(descent_direction, Hd)) + 1e-8)
        new_para = self.line_search(states, actions, advantage, old_log_probs, old_action_dists, descent_direction*max_coef)
        torch.nn.utils.convert_parameters.vector_to_parameters(new_para, self.actor.parameters()) 

    def update(self, transition_dict):
        states = transition_dict['states']
        if isinstance(states, tuple):
            states = states[0]
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                      td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()
        old_action_dists = torch.distributions.Categorical(
            self.actor(states).detach())
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()  # 更新价值函数
        self.critic_loss_list.append(critic_loss.item())
        # 更新策略函数
        self.policy_learn(states, actions, old_action_dists, old_log_probs,
                          advantage)
        
    def max_q_value(self, state):
        if type(state) is tuple:
            state = state[0]
        state = torch.tensor([state], dtype=torch.float32)
        return self.critic(state).max().item()