import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np

# ============= 1. 策略网络和价值网络 =============
class PolicyNetwork(nn.Module):
    """策略网络 π_θ(a|s)"""
    def __init__(self, state_dim, action_dim, hidden_dim=64, continuous=False):
        super(PolicyNetwork, self).__init__()
        self.continuous = continuous
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        if continuous:
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.tanh(self.fc1(state))
        x = F.tanh(self.fc2(x))
        
        if self.continuous:
            mean = self.mean(x)
            std = torch.exp(self.log_std)
            return mean, std
        else:
            logits = self.fc3(x)
            return logits
    
    def get_distribution(self, state):
        """获取动作分布"""
        if self.continuous:
            mean, std = self.forward(state)
            return Normal(mean, std)
        else:
            logits = self.forward(state)
            return Categorical(logits=logits)
    
    def get_action(self, state):
        """采样动作"""
        dist = self.get_distribution(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        if self.continuous:
            log_prob = log_prob.sum(dim=-1)
        
        return action, log_prob


class ValueNetwork(nn.Module):
    """价值网络 V(s)"""
    def __init__(self, state_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        x = F.tanh(self.fc1(state))
        x = F.tanh(self.fc2(x))
        value = self.fc3(x)
        return value.squeeze(-1)
    

# ============= 2. 优势函数估计 =============
def compute_gae(rewards, values, next_values, dones, gamma=0.99, lambda_=0.95):
    """
    计算广义优势估计 (Generalized Advantage Estimation)
    A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2 δ_{t+2} + ...
    其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)
    
    Args:
        rewards: shape (T,)
        values: shape (T,)
        next_values: shape (T,)
        dones: shape (T,)
    
    Returns:
        advantages: shape (T,)
        returns: shape (T,)
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    returns = torch.zeros(T)
    
    gae = 0
    for t in reversed(range(T)):
        # TD error: δ_t = r_t + γV(s_{t+1})(1-done) - V(s_t)
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        
        # GAE: A_t = δ_t + (γλ)(1-done)A_{t+1}
        gae = delta + gamma * lambda_ * (1 - dones[t]) * gae
        advantages[t] = gae
        
        # Return: R_t = A_t + V(s_t)
        returns[t] = advantages[t] + values[t]
    
    return advantages, returns


# ============= 3. KL散度计算 =============
def compute_kl_divergence(policy, states, old_policy_params, continuous=False):
    """
    计算新旧策略之间的KL散度
    KL(π_old || π_new) = E[log π_old(a|s) - log π_new(a|s)]
    
    对于离散动作：KL = Σ π_old(a|s) log(π_old(a|s) / π_new(a|s))
    对于连续动作：KL = log(σ_new/σ_old) + (σ_old^2 + (μ_old-μ_new)^2)/(2σ_new^2) - 1/2
    """
    if continuous:
        mean, std = policy(states)
        old_mean, old_std = old_policy_params
        
        # KL散度（高斯分布）
        kl = torch.log(std / old_std) + (old_std.pow(2) + (old_mean - mean).pow(2)) / (2 * std.pow(2)) - 0.5
        return kl.sum(dim=-1).mean()
    else:
        logits = policy(states)
        old_logits = old_policy_params
        
        # KL散度（分类分布）
        old_probs = F.softmax(old_logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        
        kl = (old_probs * (torch.log(old_probs + 1e-8) - torch.log(probs + 1e-8))).sum(dim=-1).mean()
        return kl


def get_flat_params(model):
    """获取模型的扁平化参数"""
    return torch.cat([param.view(-1) for param in model.parameters()])


def set_flat_params(model, flat_params):
    """设置模型的扁平化参数"""
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

# ============= 4. Fisher向量积 (Fisher Vector Product) =============
def fisher_vector_product(policy, states, vector, damping=1e-2, continuous=False):
    """
    计算 Fisher信息矩阵与向量的乘积: F * v
    F = E[∇log π(a|s) ∇log π(a|s)^T]
    
    使用技巧：F*v = ∇[(∇KL)^T * v]
    避免显式计算Fisher矩阵（可能非常大）
    
    Args:
        policy: 策略网络
        states: 状态
        vector: 要相乘的向量
        damping: 阻尼系数，用于数值稳定性
    
    Returns:
        fvp: Fisher向量积
    """
    # 1. 计算KL散度
    if continuous:
        mean, std = policy(states)
        old_mean = mean.detach()
        old_std = std.detach()
        
        kl = torch.log(std / old_std) + (old_std.pow(2) + (old_mean - mean).pow(2)) / (2 * std.pow(2)) - 0.5
        kl = kl.sum(dim=-1).mean()
    else:
        logits = policy(states)
        old_logits = logits.detach()
        
        old_probs = F.softmax(old_logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        
        kl = (old_probs * (torch.log(old_probs + 1e-8) - torch.log(probs + 1e-8))).sum(dim=-1).mean()
    
    # 2. 计算KL的梯度
    grads = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
    flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
    
    # 3. 计算 (∇KL)^T * v
    grad_kl_v = (flat_grad_kl * vector).sum()
    
    # 4. 计算 ∇[(∇KL)^T * v] = F * v
    grads_grad_kl_v = torch.autograd.grad(grad_kl_v, policy.parameters())
    flat_fvp = torch.cat([grad.contiguous().view(-1) for grad in grads_grad_kl_v])
    
    # 5. 添加阻尼：F * v + damping * v
    return flat_fvp + damping * vector

# ============= 5. 共轭梯度法求解 F^{-1}g =============
def conjugate_gradient(fvp_fn, b, max_iter=10, tol=1e-10):
    """
    使用共轭梯度法求解 Ax = b，其中A = Fisher矩阵
    返回 x = A^{-1}b = F^{-1}g
    
    Args:
        fvp_fn: Fisher向量积函数 (x -> Ax)
        b: 右侧向量 (策略梯度 g)
        max_iter: 最大迭代次数
        tol: 容差
    
    Returns:
        x: 解向量 F^{-1}g
    """
    x = torch.zeros_like(b)
    r = b.clone()  # 残差
    p = r.clone()  # 搜索方向
    rs_old = torch.dot(r, r)
    
    for i in range(max_iter):
        Ap = fvp_fn(p)
        alpha = rs_old / (torch.dot(p, Ap) + 1e-8)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        
        if torch.sqrt(rs_new) < tol:
            break
        
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    
    return x

# ============= 6. 线搜索 (Line Search) =============
def line_search(policy, get_loss, get_kl, old_params, fullstep, 
                max_backtracks=10, accept_ratio=0.1, max_kl=0.01):
    """
    回溯线搜索，确保满足KL约束且目标改善
    
    Args:
        policy: 策略网络
        get_loss: 计算目标函数的函数
        get_kl: 计算KL散度的函数
        old_params: 旧参数
        fullstep: 完整步长 √(2δ/(g^T F^{-1}g)) * F^{-1}g
        max_backtracks: 最大回溯次数
        accept_ratio: 接受比率
        max_kl: 最大KL散度
    
    Returns:
        success: 是否成功
    """
    old_loss = get_loss().item()
    
    for stepfrac in [0.5**i for i in range(max_backtracks)]:
        # 尝试新参数
        new_params = old_params + stepfrac * fullstep
        set_flat_params(policy, new_params)
        
        # 检查KL约束
        kl = get_kl()
        new_loss = get_loss().item()
        
        # 检查是否满足条件
        actual_improve = old_loss - new_loss
        expected_improve = stepfrac * (old_loss - new_loss)
        ratio = actual_improve / (expected_improve + 1e-8)
        
        if kl.item() <= max_kl and ratio > accept_ratio:
            print(f"  Line search success at step fraction {stepfrac:.4f}")
            print(f"  KL: {kl.item():.6f}, Loss improve: {actual_improve:.6f}")
            return True
    
    # 恢复旧参数
    set_flat_params(policy, old_params)
    print("  Line search failed, keeping old parameters")
    return False


class TRPO:

    def __init__(self, state_dim, action_dim, device, continuous=False,
                 hidden_dim=64, gamma=0.99, lambda_=0.95,
                 max_kl=0.01, damping=1e-2, cg_iters=10,
                 value_lr=1e-3):
        """
        TRPO算法
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            continuous: 是否连续动作空间
            hidden_dim: 隐藏层维度
            gamma: 折扣因子
            lambda_: GAE参数
            max_kl: 最大KL散度约束 (对应公式中的δ)
            damping: Fisher矩阵阻尼
            cg_iters: 共轭梯度迭代次数
            value_lr: 价值网络学习率
        """
        self.continuous = continuous
        self.gamma = gamma
        self.lambda_ = lambda_
        self.max_kl = max_kl
        self.damping = damping
        self.cg_iters = cg_iters
        self.device = device
        
        # 初始化网络
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim, continuous).to(self.device)
        self.value = ValueNetwork(state_dim, hidden_dim).to(self.device)
        
        # 价值网络优化器
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=value_lr)
        self.critic_loss_list = []


    def take_action(self, state):
        if type(state) is tuple:
            state = state[0]
        state = torch.tensor([state], dtype=torch.float32).to(self.device)
        probs = self.actor(state)
        action_list = torch.distributions.Categorical(probs)
        action = action_list.sample()
        return action.item()

    def select_action(self, state):
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob = self.policy.get_action(state_tensor)
            value = self.value(state_tensor)
        
        return action.numpy()[0], log_prob.item(), value.item()
    
    
    def update(self, trajectories):
        """
        TRPO更新
        
        Args:
            trajectories: 轨迹数据，格式为 [(s, a, r, s', done, log_prob, value), ...]
        
        Returns:
            info: 训练信息字典
        """
        # 1. 准备数据
        states, actions, rewards, next_states, dones, old_log_probs, values = zip(*trajectories)
        
        states = torch.FloatTensor(states)
        if self.continuous:
            actions = torch.FloatTensor(actions)
        else:
            actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        old_log_probs = torch.FloatTensor(old_log_probs)
        values = torch.FloatTensor(values)
        
        # 2. 计算next_values
        with torch.no_grad():
            next_values = self.value(next_states)
        
        # 3. 计算优势函数和回报
        advantages, returns = compute_gae(rewards, values, next_values, dones, 
                                          self.gamma, self.lambda_)
        
        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 4. 计算策略梯度 g
        dist = self.policy.get_distribution(states)
        if self.continuous:
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
        else:
            new_log_probs = dist.log_prob(actions)
        
        # 目标函数: L(θ) = E[π_new/π_old * A]
        ratio = torch.exp(new_log_probs - old_log_probs)
        surrogate_loss = -(ratio * advantages).mean()
        
        # 计算梯度 g = ∇L(θ)
        policy_grads = torch.autograd.grad(surrogate_loss, self.policy.parameters())
        policy_gradient = torch.cat([grad.view(-1) for grad in policy_grads]).detach()
        
        # 5. 使用共轭梯度求解 F^{-1}g
        def fvp_fn(v):
            return fisher_vector_product(self.policy, states, v, self.damping, self.continuous)
        
        stepdir = conjugate_gradient(fvp_fn, policy_gradient, max_iter=self.cg_iters)
        
        # 6. 计算步长 α = √(2δ / (g^T F^{-1}g))
        shs = 0.5 * torch.dot(stepdir, fvp_fn(stepdir))  # s^T H s / 2 = g^T F^{-1}g / 2
        lagrange_multiplier = torch.sqrt(shs / self.max_kl)  # √(g^T F^{-1}g / 2δ)
        fullstep = stepdir / lagrange_multiplier  # √(2δ / g^T F^{-1}g) * F^{-1}g
        
        print(f"\n  Policy gradient norm: {policy_gradient.norm().item():.6f}")
        print(f"  Step direction norm: {stepdir.norm().item():.6f}")
        print(f"  Lagrange multiplier: {lagrange_multiplier.item():.6f}")
        
        # 7. 线搜索更新参数
        old_params = get_flat_params(self.policy)
        
        # 保存旧策略参数（用于KL计算）
        if self.continuous:
            with torch.no_grad():
                old_mean, old_std = self.policy(states)
                old_policy_params = (old_mean.clone(), old_std.clone())
        else:
            with torch.no_grad():
                old_policy_params = self.policy(states).clone()
        
        def get_loss():
            dist = self.policy.get_distribution(states)
            if self.continuous:
                log_probs = dist.log_prob(actions).sum(dim=-1)
            else:
                log_probs = dist.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            return -(ratio * advantages).mean()
        
        def get_kl():
            return compute_kl_divergence(self.policy, states, old_policy_params, self.continuous)
        
        success = line_search(self.policy, get_loss, get_kl, old_params, fullstep,
                             max_kl=self.max_kl)
        
        # 8. 更新价值网络
        for _ in range(5):  # 多次更新价值网络
            value_pred = self.value(states)
            value_loss = F.mse_loss(value_pred, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            self.critic_loss_list.append(value_loss.item())
        
        # 9. 返回训练信息
        with torch.no_grad():
            final_kl = get_kl().item()
            final_loss = get_loss().item()
        
        info = {
            'policy_loss': surrogate_loss.item(),
            'value_loss': value_loss.item(),
            'kl_divergence': final_kl,
            'policy_updated': success,
            'avg_advantage': advantages.mean().item(),
            'avg_return': returns.mean().item()
        }
        
        return info
    
    def max_q_value(self, state):
        if type(state) is tuple:
            state = state[0]
        state = torch.tensor([state], dtype=torch.float32)
        return self.value(state).max().item()