"""HashNet: 连续化深度哈希（Deep Learning to Hash by Continuation）。

参考：
    Cao et al. (ICCV 2017). "HashNet: Deep Learning to Hash by Continuation."

    原始 HashNet 容易导致所有比特高度相关（比特坍缩），本实现新增：
    1. 批归一化（BatchNorm），稳定输入分布
    2. 权重正交正则化，鼓励不同比特的输出权重彼此正交，增强比特多样性
    3. 比特去相关损失，直接惩罚批次内编码的协方差矩阵的非对角线元素
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 模型定义

class HashNetModel(nn.Module):
    """带批归一化的全连接哈希模型。

    批归一化确保输入到 fc 层的特征均值为 0、方差为 1，
    使得不同比特的权重向量之间的正交约束更加有效。
    """

    def __init__(self, in_dim, n_bits):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim, affine=False, track_running_stats=True)
        self.fc = nn.Linear(in_dim, n_bits)

    def forward(self, x):
        return self.fc(self.bn(x))

# 损失函数辅助

def _pairwise_similarity(labels):
    # 构建硬二值的成对相似度矩阵。

    n = len(labels)
    S = torch.zeros(n, n, device=labels.device)
    for i in range(n):
        S[i] = (labels[i] == labels).float() * 2 - 1
    return S


def _bit_decorrelation_loss(codes):
    # 比特去相关损失，惩罚编码矩阵中各比特之间的相关性。
    B, K = codes.shape
    # 中心化
    codes_c = codes - codes.mean(dim=0, keepdim=True)
    # 协方差矩阵  (K, K)
    C = (codes_c.T @ codes_c) / (B - 1)
    # 非对角线元素
    off_diag = C - torch.diag(torch.diag(C))
    loss_d = (off_diag ** 2).sum() / max(K * (K - 1), 1)
    return loss_d


def _weight_orthogonality_loss(weight):
    # 权重正交正则化， 鼓励 fc 层权重各行彼此正交。
    K = weight.shape[0]
    WWT = weight @ weight.T                            # (K, K)
    target = torch.eye(K, device=weight.device)
    # 按行范数缩放目标，避免惩罚权重大小
    row_norms = torch.norm(weight, dim=1, keepdim=True)  # (K, 1)
    scale = row_norms @ row_norms.T                      # (K, K)
    target = target * scale
    loss_w = ((WWT - target) ** 2).sum() / (K * K)
    return loss_w


def beta_schedule(epoch, total_epochs, beta0=0.1, beta_max=20.0):
    # 连续化 β 调度函数。
    return beta0 * (beta_max / beta0) ** (epoch / max(total_epochs - 1, 1))

# 训练函数
def train_hashnet(features, labels, n_bits,
                  lr=1e-3, epochs=30, batch_size=128,
                  beta0=0.1, beta_max=20.0,
                  lambda_q=0.01, lambda_d=0.1, lambda_w=0.01,
                  device='cuda'):
    # 训练 HashNet 模型并返回所有样本的 {0,1} 哈希码。

    feats = torch.FloatTensor(features).to(device)
    labs = torch.LongTensor(labels).to(device)

    model = HashNetModel(feats.shape[1], n_bits).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(feats, labs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        beta_t = beta_schedule(epoch, epochs, beta0, beta_max)
        total_loss = 0

        for bx, bl in loader:
            codes = model(bx)
            code_norm = torch.tanh(beta_t * codes)       # 使用当前轮数的 β

            # 成对相似度损失
            S = _pairwise_similarity(bl)
            sim_pred = code_norm @ code_norm.T / n_bits
            loss_sim = -torch.mean(S * sim_pred)

            # 量化损失
            loss_q = torch.mean(torch.abs(torch.abs(code_norm) - 1))

            # 比特去相关损失
            loss_d = _bit_decorrelation_loss(code_norm)

            # 权重正交正则化
            loss_w = _weight_orthogonality_loss(model.fc.weight)

            loss = loss_sim + lambda_q * loss_q + lambda_d * loss_d + lambda_w * loss_w

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  HashNet epoch {epoch+1}/{epochs}, "
                  f"β={beta_t:.3f}, loss={total_loss/len(loader):.4f}")

    # 使用最大 β 做最终二值化
    model.eval()
    with torch.no_grad():
        all_codes = torch.tanh(beta_max * model(feats))
        binary = (all_codes > 0).float().cpu().numpy()
    return model, binary

# 编码接口

def encode(model, features, beta=20.0, device='cuda'):
    # 用训练好的 HashNet 模型将特征编码为 {0,1} 二值哈希码。

    model.eval()
    feats = torch.FloatTensor(features).to(device)
    with torch.no_grad():
        codes = torch.tanh(beta * model(feats))
    return (codes > 0).float().cpu().numpy()
