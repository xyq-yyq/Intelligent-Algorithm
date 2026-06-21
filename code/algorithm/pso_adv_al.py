"""BPSO（二进制粒子群优化）包括惯性权重衰减 + 重启 + 多样性增强。
三种机制协同：
    1. 惯性权重衰减：w_start → w_end
    2. 重启策略：适应度停滞时，保留全局最优，完全随机重置其余粒子
    3. 多样性增强：种群过于相似时，以 gBest 为基准扰动部分粒子
"""

import numpy as np
# BPSO预定义相关参数（包含惯性权重衰减、重启策略、多样性保持机制等）
_PSO_POP_SIZE = 40
_PSO_GENERATIONS = 100
_PSO_C1 = 2.0
_PSO_C2 = 2.0
_PSO_V_MAX = 6.0
_IWD_W_START = 0.9
_IWD_W_END = 0.4
_RESTART_PATIENCE = 15
_DIVERSITY_THRESHOLD = 6.0
_DIVERSITY_RESET_RATIO = 0.5

def _repair(x, ld):
    x = x.copy()
    excess = int(x.sum()) - ld
    if excess > 0:
        ones = np.where(x == 1)[0]
        x[np.random.choice(ones, excess, replace=False)] = 0
    elif excess < 0:
        zeros = np.where(x == 0)[0]
        x[np.random.choice(zeros, -excess, replace=False)] = 1
    return x

def _perturb_from_gbest(g_best, lp, ld, flip_ratio=0.15):
    new_pos = g_best.copy()
    n_flip = max(1, int(lp * flip_ratio))
    new_pos[np.random.choice(lp, n_flip, replace=False)] ^= 1
    return _repair(new_pos, ld)

def select_bits(A_hat, ld, pop_size=_PSO_POP_SIZE, generations=_PSO_GENERATIONS,
                w_start=_IWD_W_START, w_end=_IWD_W_END, c1=_PSO_C1, c2=_PSO_C2,
                v_max=_PSO_V_MAX, verbose=True, return_history=False):
    lp = A_hat.shape[0]
    thr_div = int(lp * _DIVERSITY_THRESHOLD / 256.0)

    # 初始化种群
    x_individual = np.zeros(lp, dtype=np.int8)
    x_individual[np.random.choice(lp, ld, replace=False)] = 1
    positions = np.array([x_individual for _ in range(pop_size)])
    velocities = np.random.uniform(-v_max, v_max, (pop_size, lp))
    fits = np.array([0.5 * (x @ A_hat @ x) for x in positions])

    p_best_pos = positions.copy()
    p_best_fit = fits.copy()

    g_idx = np.argmax(fits)
    g_best_pos = positions[g_idx].copy()
    g_best_fit = fits[g_idx]

    no_improve = 0
    last_best = g_best_fit
    history = [g_best_fit]

    # 主循环
    for gen in range(generations):
        w = w_start - (w_start - w_end) * (gen / generations)

        for i in range(pop_size):
            v = w * velocities[i] + c1 * np.random.random() * (p_best_pos[i] - positions[i]) + c2 * np.random.random() * (g_best_pos - positions[i])
            v = np.clip(v, -v_max, v_max)
            prob = 1.0 / (1.0 + np.exp(-v))

            new_x = (np.random.random(len(prob)) < prob).astype(np.int8)
            new_x = _repair(new_x, ld)
            new_f = 0.5 * (new_x @ A_hat @ new_x)

            positions[i] = new_x
            velocities[i] = v
            fits[i] = new_f

            if new_f > p_best_fit[i]:
                p_best_pos[i] = new_x.copy()
                p_best_fit[i] = new_f
                if new_f > g_best_fit:
                    g_best_pos = new_x.copy()
                    g_best_fit = new_f

        # 重启策略
        if g_best_fit > last_best + 1e-8:
            last_best = g_best_fit
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= _RESTART_PATIENCE:
            g_idx = np.argmax(fits)
            for i in range(pop_size):
                if i != g_idx:
                    x = np.zeros(lp, dtype=np.int8)
                    x[np.random.choice(lp, ld, replace=False)] = 1
                    positions[i] = x
                    velocities[i] = np.random.uniform(-v_max, v_max, lp)
                    fits[i] = 0.5 * (positions[i] @ A_hat @ positions[i])
                    p_best_pos[i] = positions[i].copy()
                    p_best_fit[i] = fits[i]
            no_improve = 0

        # 多样性增强
        diversity = np.mean([np.sum(p != g_best_pos) for p in positions])
        if diversity < thr_div:
            n_reset = int(pop_size * _DIVERSITY_RESET_RATIO)
            reset_idx = np.random.choice(pop_size, n_reset, replace=False)
            for idx in reset_idx:
                new_pos = _perturb_from_gbest(g_best_pos, lp, ld)
                positions[idx] = new_pos
                velocities[idx] = np.random.uniform(-v_max/2, v_max/2, lp)
                fits[idx] = 0.5 * (new_pos @ A_hat @ new_pos)
                p_best_pos[idx] = new_pos.copy()
                p_best_fit[idx] = fits[idx]

        history.append(g_best_fit)

        if verbose and (gen + 1) % 50 == 0:
            print(f"  BPSO gen {gen+1}/{generations}, best_f={g_best_fit:.4f}, "
                  f"w={w:.3f}")

    if return_history:
        return np.where(g_best_pos == 1)[0], g_best_fit, history
    return np.where(g_best_pos == 1)[0], g_best_fit
