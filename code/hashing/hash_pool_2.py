"""传统哈希池构建 —— 整合 KLSH、PCAH、LSH 三种经典哈希方法。
三种方法各提供 BITS_PER_METHOD 比特：
    比特   0– 71 : KLSH  (核化局部敏感哈希)
    比特  72–143 : PCAH  (PCA 哈希)
    比特 144–215 : LSH   (局部敏感哈希)
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from hashing.hash_pool import load_dataset, extract_features
from hashing.KLSH import train_klsh, encode as encode_klsh
from hashing.PCAH import train_pcah, encode as encode_pcah
from hashing.LSH import train_lsh, encode as encode_lsh

os.makedirs(POOL_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 三种方法，每方法 BITS_PER_METHOD 比特
_POOL_BITS = BITS_PER_METHOD * 3  # 216


def build_tradition_hash_pool():
    # 从头构建 216 位传统哈希池（KLSH + PCAH + LSH）。

    sys.stdout.reconfigure(line_buffering=True)
    print(f"构建传统哈希池，设备: {DEVICE}")
    print(f"数据集: {DATASET}, 池大小: {_POOL_BITS} 比特 (3 方法 × {BITS_PER_METHOD})")
    print(f"方法: KLSH(0-{BITS_PER_METHOD-1}) + PCAH({BITS_PER_METHOD}-{BITS_PER_METHOD*2-1}) + LSH({BITS_PER_METHOD*2}-{_POOL_BITS-1})")

    #尝试复用原始池的特征（保证 LSH 比特完全一致）
    original_pool_path = os.path.join(POOL_DIR, 'hash_pool.npz')
    if os.path.exists(original_pool_path):
        print("检测到原始哈希池，复用其特征（确保 LSH 比特一致）...")
        original = np.load(original_pool_path)
        train_feat = original['train_feat']
        test_feat = original['test_feat']
        train_labs = original['train_labels']
        test_labs = original['test_labels']
        print(f"训练特征: {train_feat.shape}, 测试特征: {test_feat.shape}")
    else:
        print("未找到原始哈希池，从零提取特征...")
        (train_paths, train_labs), (test_paths, test_labs) = load_dataset()
        print(f"训练集: {len(train_labs)}, 测试集: {len(test_labs)}")
        print("\n提取 ResNet-50 特征")
        train_feat, train_labs = extract_features(train_paths, train_labs, 'train')
        test_feat, test_labs = extract_features(test_paths, test_labs, 'test')
        print(f"训练特征: {train_feat.shape}, 测试特征: {test_feat.shape}")

    # 生成三种哈希方法（各 72 比特）
    print(f"\n--- 生成传统哈希方法（每种 {BITS_PER_METHOD} 比特，共 3 种）")

    # [1/3] KLSH
    print("\n[1/3] KLSH (核化局部敏感哈希)...")
    model_klsh, codes_klsh_train = train_klsh(
        train_feat, train_labs, BITS_PER_METHOD, seed=42)
    codes_klsh_test = encode_klsh(model_klsh, test_feat)
    np.savez_compressed(os.path.join(POOL_DIR, 'KLSH_model.npz'),
                        anchors=model_klsh["anchors"], R=model_klsh["R"],
                        gamma=model_klsh["gamma"], kernel_mean=model_klsh["kernel_mean"],
                        n_bits=model_klsh["n_bits"], seed=model_klsh["seed"])

    # [2/3] PCAH
    print("\n[2/3] PCAH (PCA 哈希)...")
    model_pcah, codes_pcah_train = train_pcah(
        train_feat, train_labs, BITS_PER_METHOD)
    codes_pcah_test = encode_pcah(model_pcah, test_feat)
    np.savez_compressed(os.path.join(POOL_DIR, 'PCAH_model.npz'),
                        W=model_pcah["W"], mean=model_pcah["mean"],
                        b=model_pcah["b"], n_bits=model_pcah["n_bits"])

    # [3/3] LSH
    print("\n[3/3] LSH (局部敏感哈希)...")
    model_lsh, codes_lsh_train = train_lsh(train_feat, train_labs, BITS_PER_METHOD, seed=42)
    codes_lsh_test = encode_lsh(model_lsh, test_feat)
    np.savez_compressed(os.path.join(POOL_DIR, 'LSH_tradition_model.npz'),
                        W=model_lsh["W"], b=model_lsh["b"],
                        seed=model_lsh["seed"], n_bits=model_lsh["n_bits"])

    # 拼接为统一哈希池
    pool_train = np.vstack([
        codes_klsh_train.T,
        codes_pcah_train.T,
        codes_lsh_train.T,
    ]).astype(np.int8)
    pool_test = np.vstack([
        codes_klsh_test.T,
        codes_pcah_test.T,
        codes_lsh_test.T,
    ]).astype(np.int8)

    print(f"\n传统哈希池形状 - 训练集: {pool_train.shape}, 测试集: {pool_test.shape}")

    # 保存
    save_path = os.path.join(POOL_DIR, 'tradition_hash_pool.npz')
    np.savez_compressed(save_path,
                        pool_train=pool_train, pool_test=pool_test,
                        train_feat=train_feat.astype(np.float32),
                        test_feat=test_feat.astype(np.float32),
                        train_labels=train_labs, test_labels=test_labs)
    print(f"\n传统哈希池已保存到: {save_path}")
    print("完成")


if __name__ == '__main__':
    build_tradition_hash_pool()
