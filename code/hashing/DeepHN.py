"""DeepHN: 深度哈希网络（Deep Hashing Network）。

参考：
    Zhu et al. (AAAI 2016). "Deep Hashing Network for Efficient Similarity Retrieval."

核心思想：
    CNN 特征 → 全连接层 → tanh(β·x) → 硬阈值二值化 → {0,1} 哈希码。
    损失函数 = 成对相似度损失（硬二值标签）+ 量化损失。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# 模型定义

class DeepHNModel(nn.Module):
    """单层全连接哈希模型：输入特征 → 线性变换 → n_bits 维输出。"""

    def __init__(self, in_dim, n_bits):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_bits)

    def forward(self, x):
        return self.fc(x)

# 损失函数辅助

def _pairwise_similarity(labels):
    n = len(labels)
    S = torch.zeros(n, n, device=labels.device)
    for i in range(n):
        S[i] = (labels[i] == labels).float() * 2 - 1   # True→1→+1, False→0→-1
    return S

# 训练函数

def train_deephn(features, labels, n_bits,
                 lr=1e-3, epochs=30, batch_size=128,
                 beta=10.0, lambda_q=0.01, device='cuda'):
    # 训练 DeepHN 模型并返回所有样本的 {0,1} 哈希码。
    feats = torch.FloatTensor(features).to(device)
    labs = torch.LongTensor(labels).to(device)
    n = feats.shape[0]

    model = DeepHNModel(feats.shape[1], n_bits).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(feats, labs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for bx, bl in loader:
            codes = model(bx)                                # (B, n_bits) 原始输出
            code_norm = torch.tanh(beta * codes)             # 近似二值化

            # 成对相似度损失
            S = _pairwise_similarity(bl)                     # (B, B) ∈ {+1, -1}
            sim_pred = code_norm @ code_norm.T / n_bits      # 归一化内积
            loss_sim = -torch.mean(S * sim_pred)

            # 量化损失：鼓励 |output| → 1
            loss_q = torch.mean(torch.abs(torch.abs(code_norm) - 1))

            loss = loss_sim + lambda_q * loss_q
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  DeepHN epoch {epoch+1}/{epochs}, "
                  f"loss={total_loss/len(loader):.4f}")

    # 生成最终哈希码
    model.eval()
    with torch.no_grad():
        all_codes = torch.tanh(beta * model(feats))
        binary = (all_codes > 0).float().cpu().numpy()       # >0 → 1, ≤0 → 0
    return model, binary


# 编码接口

def encode(model, features, beta=10.0, device='cuda'):
    # 用训练好的 DeepHN 模型将特征编码为 {0,1} 二值哈希码。
    model.eval()
    feats = torch.FloatTensor(features).to(device)
    with torch.no_grad():
        codes = torch.tanh(beta * model(feats))
    return (codes > 0).float().cpu().numpy()
