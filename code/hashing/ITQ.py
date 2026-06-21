"""ITQ (迭代量化) — Iterative Quantization.

参考：
    Gong & Lazebnik (CVPR 2011). "Iterative Quantization: A Procrustean
    Approach to Learning Binary Codes."

ITQ 是 PCAH 的改进：通过旋转使数据更好地适应超立方体顶点，
显著减少量化误差，在多个基准上优于 PCAH 和 LSH。
"""

import numpy as np


def train_itq(features, labels, n_bits, n_iters=50, seed=42, **kwargs):
    # 训练 ITQ 模型。

    rng = np.random.RandomState(seed)
    n, d = features.shape
    n_bits = min(n_bits, d)

    # 数据中心化 + PCA
    mean = features.mean(axis=0, keepdims=True)
    Xc = features - mean

    # SVD: Xc = U Σ Vᵀ
    U_full, S_full, Vt_full = np.linalg.svd(Xc, full_matrices=False)
    W_pca = Vt_full[:n_bits].T                              # (d, n_bits)
    V = Xc @ W_pca                                           # (n, n_bits) — PCA 投影

    # 初始化随机正交旋转 R
    R = rng.randn(n_bits, n_bits)
    U_r, _, Vt_r = np.linalg.svd(R)
    R = U_r @ Vt_r                                           # 正交矩阵

    # 交替优化
    for it in range(n_iters):
        # Fix R, update B
        B = np.sign(V @ R)                                   # (n, n_bits), ∈ {+1, -1}
        B[B == 0] = 1                                        # 处理零值

        # Fix B, update R via Procrustes: B'V = U Σ V', R = V U'
        C = B.T @ V                                          # (n_bits, n_bits)
        Ub, _, Vtb = np.linalg.svd(C)
        R = Vtb.T @ Ub.T                                     # 正交

    # 最终编码
    projections = V @ R                                      # (n, n_bits)
    b = np.median(projections, axis=0)
    train_codes = (projections > b).astype(np.int8)

    bit_means = train_codes.mean(axis=0)
    dead = np.sum((bit_means < 0.05) | (bit_means > 0.95))
    print(f"  ITQ: {n_bits} bits, iters={n_iters}, "
          f"bit balance={bit_means.mean():.3f} (理想=0.5), 死比特={dead}")

    model = {
        "W": W_pca,        # (d, n_bits) PCA 投影矩阵
        "R": R,            # (n_bits, n_bits) 最优旋转矩阵
        "mean": mean,      # (1, d) 数据中心
        "b": b,            # (n_bits,) 中位数阈值
        "n_bits": n_bits,
        "seed": seed,
    }
    return model, train_codes


def encode(model, features, **kwargs):
    # 用训练好的 ITQ 模型编码新特征。

    Xc = features - model["mean"]
    V = Xc @ model["W"]
    projections = V @ model["R"]
    return (projections > model["b"]).astype(np.int8)
