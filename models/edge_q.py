# edge_q.py
# -*- coding: utf-8 -*-
"""
Dueling DQN 和 普通DQN 的边级 Q 网络：
- 输入：每条候选边的特征 x_e（如 [1/delay, rem_vis, conn_gain, is_new, layer_u, layer_v]）
- 输出：每条边的 Q 值
"""
import math, torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeQNet(nn.Module):
    '''
    Dueling DQN
    '''
    def __init__(self, edge_dim:int, hidden:int=256):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)   # per-edge advantage
        )
        self.val_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)   # global value（通过边表征的均值近似）
        )

    def forward(self, x_e:torch.Tensor):
        if x_e.numel()==0:
            return x_e
        h = self.edge_mlp(x_e)                # [E, H]
        ctx = h.mean(dim=0, keepdim=True)     # [1, H]
        V = self.val_head(ctx)                # [1,1]
        A = self.adv_head(h)                  # [E,1]
        Q = V + A - A.mean(dim=0, keepdim=True)
        return Q.squeeze(-1)                  # [E]


#---------------标准DQN---------------------------------

class EdgeDQNet(nn.Module):
    """
    普通 DQN 的边级 Q 网络：
    - 输入：每条候选边的特征 x_e（形状 [E, edge_dim]）
    - 输出：每条边对应的 Q 值（形状 [E]）
    """
    def __init__(self, edge_dim: int, hidden: int = 256):
        super().__init__()
        # 直接用一个 MLP 从 edge feature 映射到标量 Q(s, a=edge)
        self.q_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)  # 直接输出每条边的 Q 值
        )

    def forward(self, x_e: torch.Tensor):
        # x_e: [E, edge_dim]
        if x_e.numel() == 0:
            return x_e
        q = self.q_mlp(x_e)           # [E, 1]
        return q.squeeze(-1)          # [E]

#----------------残差DQN--------------------------------
class _ResBlock(nn.Module):
    def __init__(self, d, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(d, d),
        )
    def forward(self, x): return x + self.net(x)

class EdgeQResMLP(nn.Module):
    '''
    在 MLP 里加 LayerNorm + 残差 + GELU，提升表达力与训练稳定性。
    '''
    def __init__(self, edge_dim:int, hidden:int=256, n_blocks:int=3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(edge_dim, hidden), nn.GELU())
        self.blocks = nn.Sequential(*[_ResBlock(hidden) for _ in range(n_blocks)])
        self.out = nn.Linear(hidden, 1)
    def forward(self, x_e:torch.Tensor):
        if x_e.numel()==0: return x_e
        h = self.inp(x_e)
        h = self.blocks(h)
        return self.out(h).squeeze(-1)


#---------------Noisy DQN---------------------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_f, out_f, sigma_init=0.5):
        super().__init__()
        self.mu_w  = nn.Parameter(torch.empty(out_f, in_f))
        self.sigma_w = nn.Parameter(torch.full((out_f, in_f), sigma_init / math.sqrt(in_f)))
        self.mu_b  = nn.Parameter(torch.empty(out_f))
        self.sigma_b = nn.Parameter(torch.full((out_f,), sigma_init / math.sqrt(out_f)))
        self.register_buffer("eps_w", torch.zeros(out_f, in_f))
        self.register_buffer("eps_b", torch.zeros(out_f))
        self.reset_parameters()
    def reset_parameters(self):
        bound = 1 / math.sqrt(self.mu_w.size(1))
        nn.init.uniform_(self.mu_w, -bound, bound)
        nn.init.uniform_(self.mu_b, -bound, bound)
    def forward(self, x):
        self.eps_w.normal_(); self.eps_b.normal_()
        w = self.mu_w + self.sigma_w * self.eps_w
        b = self.mu_b + self.sigma_b * self.eps_b
        return x @ w.t() + b

class EdgeQNoisy(nn.Module):
    '''
    把 Linear 换成 NoisyLinear；损失不变，动作分数会带可学习噪声（训练时启用）。
    '''
    def __init__(self, edge_dim:int, hidden:int=256):
        super().__init__()
        self.net = nn.Sequential(
            NoisyLinear(edge_dim, hidden), nn.ReLU(),
            NoisyLinear(hidden, hidden), nn.ReLU(),
            NoisyLinear(hidden, 1),
        )
    def forward(self, x_e):
        if x_e.numel()==0: return x_e
        return self.net(x_e).squeeze(-1)


#---------------Attention DQN---------------------------------

class EdgeQAttnCtx(nn.Module):
    '''
    注意力上下文池化（替代 mean，更聪明的“全局上下文”）
    '''
    def __init__(self, edge_dim:int, hidden:int=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(edge_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.score = nn.Linear(hidden, 1)              # 注意力打分
        self.head  = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x_e:torch.Tensor):
        if x_e.numel()==0: return x_e
        h = self.enc(x_e)                               # [E,H]
        w = F.softmax(self.score(h).squeeze(-1), dim=0) # [E]
        ctx = (w.unsqueeze(-1) * h).sum(dim=0, keepdim=True)  # [1,H]
        h_cat = torch.cat([h, ctx.expand_as(h)], dim=-1)      # [E,2H]
        return self.head(h_cat).squeeze(-1)
