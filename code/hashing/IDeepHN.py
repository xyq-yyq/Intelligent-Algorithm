"""IDeepHN: 改进的深度哈希网络（Improved Deep Hashing Network）。

参考：
    Zhang et al. (IEEE TMM 2020). "Improved Deep Hashing with Soft Pairwise
    Similarity for Large-Scale Image Retrieval."
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 模型定义

class IDeepHNModel(nn.Module):
    # 带批归一化的全连接哈希模型。

    def __init__(self, in_dim, n_bits):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim, affine=False, track_running_stats=True)
        self.fc = nn.Linear(in_dim, n_bits)

    def forward(self, x):
        return self.fc(self.bn(x))

# 软相似度

def _batch_soft_similarity(feats, labels, alpha=0.7):
    # 在批次内在线计算软相似度矩阵。

    # 标签相似度：同类 → +1，不同类 → -1
    label_sim = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() * 2 - 1

    # 特征余弦相似度：已在 [-1, 1] 范围内
    feats_norm = F.normalize(feats, p=2, dim=1)
    cos_sim = feats_norm @ feats_norm.T

    return alpha * label_sim + (1 - alpha) * cos_sim


def _bit_decorrelation_loss(codes):
    # 比特去相关损失，惩罚编码矩阵中各比特之间的相关性。

    B, K = codes.shape
    codes_c = codes - codes.mean(dim=0, keepdim=True)
    C = (codes_c.T @ codes_c) / (B - 1)
    off_diag = C - torch.diag(torch.diag(C))
    return (off_diag ** 2).sum() / max(K * (K - 1), 1)


def _weight_orthogonality_loss(weight):
    # 权重正交正则化，鼓励 fc 层权重各行彼此正交。

    K = weight.shape[0]
    WWT = weight @ weight.T
    # 按行范数缩放目标对角矩阵
    row_norms = torch.norm(weight, dim=1, keepdim=True)
    target = torch.diag_embed((row_norms ** 2).squeeze())
    return ((WWT - target) ** 2).sum() / (K * K)

# 训练函数

def train_ideephn(features, labels, n_bits,
                  lr=1e-3, epochs=30, batch_size=128,
                  beta=10.0, alpha=0.7,
                  lambda_q=0.01, lambda_d=0.1, lambda_w=0.01,
                  device='cuda'):
    # 训练 IDeepHN 模型并返回所有样本的 {0,1} 哈希码。

    feats = torch.FloatTensor(features).to(device)
    labs = torch.LongTensor(labels).to(device)

    model = IDeepHNModel(feats.shape[1], n_bits).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(feats, labs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"  使用批次内软相似度训练 (α={alpha})")
    for epoch in range(epochs):
        total_loss = 0
        for bx, bl in loader:
            codes = model(bx)
            code_norm = torch.tanh(beta * codes)

            # 在线计算批次内软相似度（无需预计算 N×N 矩阵）
            S_batch = _batch_soft_similarity(bx, bl, alpha)

            # 软相似度损失
            sim_pred = code_norm @ code_norm.T / n_bits
            loss_sim = -torch.mean(S_batch * sim_pred)

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
            print(f"  IDeepHN epoch {epoch+1}/{epochs}, "
                  f"loss={total_loss/len(loader):.4f}")

    # 生成最终哈希码
    model.eval()
    with torch.no_grad():
        all_codes = torch.tanh(beta * model(feats))
        binary = (all_codes > 0).float().cpu().numpy()
    return model, binary

# 编码接口

def encode(model, features, beta=10.0, device='cuda'):
    # 用训练好的 IDeepHN 模型将特征编码为 {0,1} 二值哈希码。

    model.eval()
    feats = torch.FloatTensor(features).to(device)
    with torch.no_grad():
        codes = torch.tanh(beta * model(feats))
    return (codes > 0).float().cpu().numpy()
