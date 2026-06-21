"""亲和度矩阵计算与检索评估指标。

核心公式来源于 Liu et al. (2017) "Hash Bit Selection for Nearest Neighbor Search"
本模块还实现了检索评估的常用指标：
    MAP, Precision@K, Recall@K, PH2
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, spdiags, diags
from sklearn.neighbors import NearestNeighbors


# 亲和度矩阵

def compute_similarity_matrix(features, k=100):
    # 构建 kNN 相似度矩阵 S（Eq.1）。

    n = features.shape[0]
    print(f"    拟合 kNN (k={k})，样本数 {n}...")
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric='euclidean')
    nn.fit(features)
    distances, indices = nn.kneighbors(features)

    # 取所有样本的第 1 到第 k 近邻距离的均值
    sigma = np.mean(distances[:, 1:]) if distances.shape[1] > 1 else 1.0
    print(f"    构建稀疏相似度矩阵 (σ={sigma:.4f})...")

    rows, cols, data = [], [], []
    for i in range(n):
        for j_idx in range(1, len(indices[i])):   # 跳过自身（第 0 个近邻）
            j = indices[i][j_idx]
            d2 = distances[i][j_idx] ** 2
            rows.append(i)
            cols.append(j)
            data.append(np.exp(-d2 / (sigma ** 2)))

    S = csr_matrix((data, (rows, cols)), shape=(n, n))
    S = S.maximum(S.T)
    return S


def compute_laplacian(S, normalized=True):
    # 计算（归一化）拉普拉斯矩阵。

    D_diag = np.asarray(S.sum(axis=1)).flatten()
    D_diag = np.maximum(D_diag, 1e-12)              # 防止除零
    D = spdiags(D_diag, 0, len(D_diag), len(D_diag), format='csr')
    L = (D - S).tocsr()
    if normalized:
        D_inv_sqrt = diags(1.0 / np.sqrt(D_diag), 0, format='csr')
        L = D_inv_sqrt @ L @ D_inv_sqrt
    return L


def compute_bit_reliability(hash_codes, L, xi=2.0):
    # 计算每个比特的可靠性 ）。
    lp, n = hash_codes.shape
    phi = np.zeros(lp)
    for k in range(lp):
        y = hash_codes[k].astype(np.float64)
        # L2 归一化使二次型尺度无关
        y_norm = y / (np.linalg.norm(y) + 1e-12)
        Ly = L @ y_norm
        val = y_norm @ Ly
        phi[k] = np.exp(-xi * val)
        if (k + 1) % 32 == 0:
            print(f"    比特可靠性: {k+1}/{lp} 完成")
    return phi


def pairwise_mutual_information(hash_codes):
    """计算所有比特对之间的互信息 I(y_i, y_j)。
    互信息衡量两个比特携带信息的重叠程度
    """
    lp, n = hash_codes.shape
    print(f"    成对互信息: lp={lp}, n={n}")
    MI = np.zeros((lp, lp))

    for i in range(lp):
        yi = hash_codes[i]
        pyi0 = np.mean(yi == 0)
        pyi1 = 1 - pyi0
        for j in range(i + 1, lp):
            yj = hash_codes[j]
            pyj0 = np.mean(yj == 0)
            pyj1 = 1 - pyj0
            mi = 0.0
            for a in [0, 1]:
                pa = pyi0 if a == 0 else pyi1
                if pa == 0:
                    continue
                for b in [0, 1]:
                    pb = pyj0 if b == 0 else pyj1
                    if pb == 0:
                        continue
                    pab = np.mean((yi == a) & (yj == b))
                    if pab > 0:
                        mi += pab * np.log(pab / (pa * pb))
            MI[i, j] = MI[j, i] = mi
        if (i + 1) % 32 == 0:
            print(f"    互信息: {i+1}/{lp} 完成")
    return MI


def compute_affinity_matrix(hash_codes, features=None, k=100, xi=2.0, beta=2.0):
    # 计算亲和度矩阵
    print(f"  计算亲和度矩阵: lp={hash_codes.shape[0]}, n={hash_codes.shape[1]}")

    # 拉普拉斯矩阵
    feats = features if features is not None else hash_codes.T.astype(np.float64)
    S = compute_similarity_matrix(feats, k)
    L = compute_laplacian(S)

    #  比特可靠性
    print("  计算比特可靠性 (φ)...")
    phi = compute_bit_reliability(hash_codes, L, xi)

    # 比特互补性 A
    print("  计算比特互补性 (成对互信息)...")
    MI = pairwise_mutual_information(hash_codes)
    A = np.exp(-beta * MI)          # 互信息越小 → 互补性越高
    np.fill_diagonal(A, 0)          # 对角线置零（比特与自身无互补概念）

    Phi = np.diag(phi)
    A_hat = Phi @ A @ Phi

    print(f"  亲和度矩阵形状: {A_hat.shape}")
    return A_hat, phi, A


# 目标函数

def objective(x, A_hat):
    # 选择向量 x 的目标函数值: f(x) = ½·xᵀ·Â·x。

    return 0.5 * (x @ A_hat @ x)

# 检索评估指标

def hamming_distance(q, db):
    # 计算单个查询向量 q 与检索库中所有向量的汉明距离。

    return np.count_nonzero(q != db, axis=1)


def compute_map(query_codes, db_codes, query_labels, db_labels, top_k=None):
    # 计算 Mean Average Precision (MAP)。

    nq = query_codes.shape[0]
    if top_k is None:
        top_k = db_codes.shape[0]
    ap_sum = 0.0
    for i in range(nq):
        dists = hamming_distance(query_codes[i], db_codes)
        order = np.argsort(dists)[:top_k]             # 按距离升序排列
        relevant = (db_labels[order] == query_labels[i])
        # 每个位置上的累计精度
        precisions = np.cumsum(relevant) / np.arange(1, len(relevant) + 1)

        ap_sum += np.sum(precisions * relevant) / max(np.sum(relevant), 1)
    return ap_sum / nq


def compute_precision_recall(query_codes, db_codes, query_labels, db_labels,
                              nk=500):
    # 计算 Precision@K 和 Recall@K。

    nq = query_codes.shape[0]
    tp_total = 0
    total_relevant = 0
    for i in range(nq):
        dists = hamming_distance(query_codes[i], db_codes)
        order = np.argsort(dists)[:nk]
        tp_total += np.sum(db_labels[order] == query_labels[i])
        total_relevant += np.sum(db_labels == query_labels[i])
    precision = tp_total / (nq * nk) if (nq * nk) > 0 else 0
    recall = tp_total / total_relevant if total_relevant > 0 else 0
    return precision, recall


def compute_ph2(query_codes, db_codes, query_labels, db_labels):
    # 计算 PH2 —— 汉明半径 2 内的检索精度。

    nq = query_codes.shape[0]
    tp, total = 0, 0
    for i in range(nq):
        dists = hamming_distance(query_codes[i], db_codes)
        mask = dists <= 2
        tp += np.sum(db_labels[mask] == query_labels[i])
        total += np.sum(mask)

    if total < 10:
        return 0.0

    return tp / total


def compute_all_metrics(query_codes, db_codes, query_labels, db_labels,
                         top_k=900):
    # 一次汉明距离计算，产出 MAP / P@K / R@K / PH2 四个指标。

    nq = query_codes.shape[0]
    ndb = db_codes.shape[0]
    nk = min(top_k, ndb)

    # 矩阵乘法一次性计算所有汉明距离 (n_q, n_db)
    q_codes = query_codes.astype(np.float32)
    d_codes = db_codes.astype(np.float32)
    q_cnt = q_codes.sum(axis=1, keepdims=True)          # (n_q, 1)
    d_cnt = d_codes.sum(axis=1)                          # (n_db,)
    all_dists = q_cnt + d_cnt[None, :] - 2 * (q_codes @ d_codes.T)
    all_dists = np.round(all_dists).astype(np.int32)     # 浮点→整数

    ap_sum = 0.0
    tp_total = 0
    total_relevant = 0
    ph2_tp, ph2_total = 0, 0

    for i in range(nq):
        dists = all_dists[i]
        order = np.argsort(dists)

        # MAP
        top_order = order[:nk]
        relevant = (db_labels[top_order] == query_labels[i]).astype(np.float64)
        precisions = np.cumsum(relevant) / np.arange(1, nk + 1)
        n_rel = relevant.sum()
        if n_rel > 0:
            ap_sum += np.sum(precisions * relevant) / n_rel

        # P@K、R@K
        tp_total += int(relevant.sum())
        total_relevant += int(np.sum(db_labels == query_labels[i]))

        # PH2
        mask = dists <= 2
        ph2_tp += int(np.sum(db_labels[mask] == query_labels[i]))
        ph2_total += int(np.sum(mask))

    map_val = ap_sum / nq
    precision = tp_total / (nq * nk) if (nq * nk) > 0 else 0.0
    recall = tp_total / total_relevant if total_relevant > 0 else 0.0
    ph2 = ph2_tp / ph2_total if ph2_total >= 10 else 0.0

    return map_val, precision, recall, ph2


def compute_h2_hits(query_codes, db_codes):
    # 计算汉明半径 2 内的平均命中数 —— 衡量哈希码的聚类紧密度。

    nq = query_codes.shape[0]
    total = 0
    for i in range(nq):
        dists = hamming_distance(query_codes[i], db_codes)
        total += int(np.sum(dists <= 2))
    return total / nq
