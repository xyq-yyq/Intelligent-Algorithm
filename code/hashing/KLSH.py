"""KLSH (核化局部敏感哈希) — Kernelized Locality-Sensitive Hashing.

参考：
    Kulis & Grauman (ICCV 2009). "Kernelized Locality-Sensitive Hashing."


    在核特征空间中构造随机投影，使 LSH 能处理非线性相似度结构。
"""

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


def train_klsh(features, labels, n_bits, n_anchors=300, gamma=None, seed=42, **kwargs):
    # 训练 KLSH 随机投影模型。

    rng = np.random.RandomState(seed)
    n, d = features.shape

    if gamma is None:
        gamma = 1.0 / d

    #  随机选取锚点
    n_anchors = min(n_anchors, n)
    anchor_idx = rng.choice(n, n_anchors, replace=False)
    anchors = features[anchor_idx].astype(np.float64)

    # 计算 RBF 核矩阵
    K = rbf_kernel(features, anchors, gamma=gamma)      # (n, m)

    # 中心化核矩阵（隐式映射到零均值特征空间）
    K_mean = K.mean(axis=0, keepdims=True)
    K_centered = K - K_mean

    #  随机投影
    R = rng.randn(n_anchors, n_bits)                     # (m, n_bits)
    projections = K_centered @ R                         # (n, n_bits)

    # 中位数阈值
    b = np.median(projections, axis=0)                   # (n_bits,)
    train_codes = (projections > b).astype(np.int8)

    bit_means = train_codes.mean(axis=0)
    dead = np.sum((bit_means < 0.05) | (bit_means > 0.95))
    print(f"  KLSH: {n_bits} bits, anchors={n_anchors}, "
          f"bit balance={bit_means.mean():.3f} (理想=0.5), 死比特={dead}")

    model = {
        "anchors": anchors,
        "R": R,
        "gamma": gamma,
        "kernel_mean": K_mean,
        "n_bits": n_bits,
        "seed": seed,
    }
    return model, train_codes


def encode(model, features, **kwargs):
    # 用训练好的 KLSH 模型编码新特征。

    K = rbf_kernel(features, model["anchors"], gamma=model["gamma"])
    K_centered = K - model["kernel_mean"]
    projections = K_centered @ model["R"]
    b = np.median(projections, axis=0)
    return (projections > b).astype(np.int8)
