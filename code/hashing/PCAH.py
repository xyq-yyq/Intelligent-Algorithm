"""PCAH (PCA哈希) — Principal Component Analysis Hashing.

参考：
    Wang et al. (2010). "Semi-Supervised Hashing for Large Scale Search."
    (PCA 方向作为哈希投影方向)
PCA 方向捕获最大数据方差方向，使哈希码携带最多的数据变化信息。

"""

import numpy as np


def train_pcah(features, labels, n_bits, **kwargs):
    # 训练 PCAH 投影模型。

    n, d = features.shape
    n_bits = min(n_bits, d)   # 比特数不能超过特征维度

    # 数据中心化
    mean = features.mean(axis=0, keepdims=True)           # (1, d)
    Xc = features - mean                                  # (n, d)

    #  SVD 分解获取主成分
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)     # Vt: (d, d), 行即特征向量
    W = Vt[:n_bits].T                                      # (d, n_bits), 前 n_bits 个主成分

    # 投影
    projections = Xc @ W                                   # (n, n_bits)

    # 中位数阈值
    b = np.median(projections, axis=0)                     # (n_bits,)
    train_codes = (projections > b).astype(np.int8)

    bit_means = train_codes.mean(axis=0)
    dead = np.sum((bit_means < 0.05) | (bit_means > 0.95))
    print(f"  PCAH: {n_bits} bits, variance_ratio={S[:n_bits].sum()/S.sum():.3f}, "
          f"bit balance={bit_means.mean():.3f} (理想=0.5), 死比特={dead}")

    model = {"W": W, "mean": mean, "b": b, "n_bits": n_bits}
    return model, train_codes


def encode(model, features, **kwargs):
    # 用训练好的 PCAH 模型编码新特征。

    Xc = features - model["mean"]
    projections = Xc @ model["W"]
    return (projections > model["b"]).astype(np.int8)
