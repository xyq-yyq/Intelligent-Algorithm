"""遗传算法（Genetic Algorithm）—— 用于哈希比特选择。
    编码：二值向量，长度为 lp，恰好包含 ld 个 1
    选择：锦标赛选择（tournament size = 3）
    交叉：单点交叉 + 基数修复
    变异：随机 1↔0 交换
"""

import numpy as np

def _mutate(x, ld, prob=0.05):
    if np.random.random() > prob:
        return x.copy()

    xm = x.copy()
    ones = np.where(xm == 1)[0]    # 所有值为 1 的位置
    zeros = np.where(xm == 0)[0]   # 所有值为 0 的位置
    if len(ones) > 0 and len(zeros) > 0:
        i = np.random.choice(ones)
        j = np.random.choice(zeros)
        xm[i], xm[j] = 0, 1       # 交换
    return xm

# GA 主循环

def run_ga(A_hat, ld,
           pop_size=100,          # 种群大小
           generations=200,       # 进化代数
           crossover_prob=0.8,    # 交叉概率
           mutation_prob=0.05,    # 变异概率
           verbose=True):

    lp = A_hat.shape[0]

    # 初始化种群
    x1 = np.zeros(lp, dtype=np.int8)
    idx = np.random.choice(lp, ld, replace=False)
    x1[idx] = 1
    pop = np.array([x1 for _ in range(pop_size)])
    fit = np.array([0.5 * (x @ A_hat @ x) for x in pop])

    # 初始最优
    best_idx = np.argmax(fit)
    best_x, best_f = pop[best_idx].copy(), fit[best_idx]
    history = [best_f]

    # 进化循环
    for gen in range(generations):
        new_pop = []

        # 生成新一代
        while len(new_pop) < pop_size:
            idx1 = np.random.choice(len(pop), 3, replace=False)
            best1 = idx1[np.argmax(fit[idx1])]
            idx2 = np.random.choice(len(pop), 3, replace=False)
            best2 = idx2[np.argmax(fit[idx2])]
            p1 = pop[best1].copy()
            p2 = pop[best2].copy()

            if np.random.random() > crossover_prob:
                new_pop.append(p1.copy())
                new_pop.append(p2.copy())
                continue

            lp = len(p1)
            # 单点交叉：随机选择一个交叉点 k
            k = np.random.randint(1, lp)
            c1 = np.concatenate([p1[:k], p2[k:]])
            c2 = np.concatenate([p2[:k], p1[k:]])

            # 修复基数约束
            for c in [c1, c2]:
                while c.sum() > ld:  # 1 太多 → 随机去掉一些
                    ones = np.where(c == 1)[0]
                    c[np.random.choice(ones)] = 0
                while c.sum() < ld:  # 1 太少 → 随机补上一些
                    zeros = np.where(c == 0)[0]
                    c[np.random.choice(zeros)] = 1

            c1 = _mutate(c1, ld, mutation_prob)
            c2 = _mutate(c2, ld, mutation_prob)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        # 更新种群
        pop = np.array(new_pop[:pop_size])
        fit = np.array([0.5 * (x @ A_hat @ x) for x in pop])

        # 更新全局最优
        cur_best_idx = np.argmax(fit)
        if fit[cur_best_idx] > best_f:
            best_x = pop[cur_best_idx].copy()
            best_f = fit[cur_best_idx]

        history.append(best_f)

        # 每 50 代打印一次进度
        if verbose and (gen + 1) % 50 == 0:
            print(f"  GA 代数 {gen+1}/{generations}, "
                  f"最优适应度={best_f:.4f}, 种群均值={fit.mean():.4f}")

    return best_x, best_f, history

# 外部接口select_bits

def select_bits(A_hat, ld, **kwargs):
    return_history = kwargs.pop('return_history', False)
    best_x, best_f, history = run_ga(A_hat, ld, **kwargs)
    selected = np.where(best_x == 1)[0]
    if return_history:
        return selected, best_f, history
    return selected, best_f
