"""使用纯随机的方式选取ld个哈希位
仅用来与其他算法做比较参考
"""

import numpy as np

def select_bits(A_hat, ld, **kwargs):
    """纯随机选择：从 lp 个比特中随机抽取 ld 个，不做任何优化。
            objective_value  : float        0.0（不做评估）
    """
    lp = A_hat.shape[0]
    idx = np.random.choice(lp, ld, replace=False)
    return idx, 0.0
