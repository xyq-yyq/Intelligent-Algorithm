"""LSH (局部敏感哈希) — 基于随机投影生成哈希码。

与 DeepHN/HashNet/IDeepHN 相同，本模块提供 train_lsh 和 encode 两个接口，
供 hash_pool.py 统一调用。LSH 无需神经网络训练，"模型"即随机投影矩阵。
"""
import numpy as np


def train_lsh(features, labels, n_bits, seed=42, **kwargs):
    # 生成 LSH 随机投影矩阵，并用中位数阈值编码训练集。

    rng = np.random.RandomState(seed)
    d = features.shape[1]

    # 随机投影矩阵
    W = rng.randn(d, n_bits)                               # (d, n_bits)

    # 计算训练集投影值
    projections = features @ W                              # (n, n_bits)

    # 中位数阈值
    # 对每个投影方向，计算训练集中位数为阈值
    # 这保证每个比特在训练集上精确 50/50 分布
    b = np.median(projections, axis=0)                     # (n_bits,)

    # 二值化
    train_codes = (projections > b).astype(np.int8)

    bit_means = train_codes.mean(axis=0)
    dead = np.sum((bit_means < 0.05) | (bit_means > 0.95))
    print(f"  LSH: {n_bits} bits, bit balance={bit_means.mean():.3f} "
          f"(理想=0.5), 死比特={dead}")

    model = {"W": W, "b": b, "seed": seed, "n_bits": n_bits}
    return model, train_codes


def encode(model, features, **kwargs):
    # 用已生成的 LSH 投影矩阵和中位数阈值编码特征。

    W = model["W"]
    b = model["b"]
    projections = features @ W
    return (projections > b).astype(np.int8)
