"""PMITS：并行文化基因迭代禁忌搜索
"""

import numpy as np
import time
# 向量化成对交换局部搜索（LS1）

def _local_search(x, A_hat, ld):
    # 向量化最陡下降爬山法：每轮批量计算所有交换的 Δ，取最优改进。

    lp = A_hat.shape[0]
    x_cur = x.copy().astype(np.int8)
    Ax = A_hat @ x_cur.astype(np.float64)

    while True:
        ones = np.where(x_cur == 1)[0]
        zeros = np.where(x_cur == 0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break

        k, m = len(ones), len(zeros)

        # 向量化计算所有 k×m 的 Δ
        # Δ(i,j) = Ax[j] - Ax[i] + 0.5*(A_ii + A_jj - 2*A_ij)
        Ax_ones = Ax[ones]                                       # (k,)
        Ax_zeros = Ax[zeros]                                     # (m,)
        A_ii = A_hat[ones, ones]                                 # (k,)
        A_jj = A_hat[zeros, zeros]                               # (m,)
        A_ij = A_hat[ones[:, None], zeros]                       # (k, m)

        deltas = (Ax_zeros[None, :] - Ax_ones[:, None]
                  + 0.5 * (A_ii[:, None] + A_jj[None, :] - 2 * A_ij))

        # 最优移动
        flat_idx = np.argmax(deltas)
        best_delta = deltas.flat[flat_idx]
        if best_delta <= 1e-12:
            break

        i_idx = flat_idx // m
        j_idx = flat_idx % m
        best_i, best_j = ones[i_idx], zeros[j_idx]

        # 执行交换
        x_cur[best_i], x_cur[best_j] = 0, 1
        Ax = Ax + A_hat[:, best_j] - A_hat[:, best_i]

    return x_cur

# 简化禁忌搜索

def _tabu_search(x0, A_hat, ld, max_no_improve=50):
    """简化禁忌搜索 + 渴望准则。
    添加渴望准则——若禁忌移动优于当前全局最优，破禁接受。
    停止条件：连续 max_no_improve 次迭代无改进。
    """
    lp = A_hat.shape[0]
    x_cur = x0.copy()
    f_cur = 0.5 * (x_cur @ A_hat @ x_cur)
    Ax = A_hat @ x_cur.astype(np.float64)

    best_x = x_cur.copy()
    best_f = f_cur

    tabu_set = {tuple(np.where(x_cur == 1)[0].tolist())}
    stall = 0

    while stall < max_no_improve:
        ones = np.where(x_cur == 1)[0]
        zeros = np.where(x_cur == 0)[0]
        k, m = len(ones), len(zeros)

        if k == 0 or m == 0:
            break

        # 向量化计算所有 Δ
        Ax_ones = Ax[ones]
        Ax_zeros = Ax[zeros]
        A_ii = A_hat[ones, ones]
        A_jj = A_hat[zeros, zeros]
        A_ij = A_hat[ones[:, None], zeros]

        deltas = (Ax_zeros[None, :] - Ax_ones[:, None]
                  + 0.5 * (A_ii[:, None] + A_jj[None, :] - 2 * A_ij))

        # 从最优到最差扫描，找第一个非禁忌移动
        sort_idx = np.argsort(-deltas.ravel())   # 降序
        best_delta = -np.inf
        best_i, best_j = -1, -1

        for idx in sort_idx:
            i_idx = idx // m
            j_idx = idx % m
            i_bit, j_bit = ones[i_idx], zeros[j_idx]

            # 渴 望准则：即使 tabu，若优于全局最优则破禁
            delta_val = deltas[i_idx, j_idx]
            if f_cur + delta_val > best_f + 1e-12:
                best_delta = delta_val
                best_i, best_j = i_bit, j_bit
                break

            # 生成键值并检查禁忌状态
            x_tmp = x_cur.copy()
            x_tmp[i_bit], x_tmp[j_bit] = 0, 1
            key = tuple(np.where(x_tmp == 1)[0].tolist())
            if key not in tabu_set:
                best_delta = delta_val
                best_i, best_j = i_bit, j_bit
                break

        if best_i == -1:    # 所有移动都禁忌且无渴望
            break

        # 执行最优非禁忌移动
        x_cur[best_i], x_cur[best_j] = 0, 1
        f_cur += best_delta
        Ax = Ax + A_hat[:, best_j] - A_hat[:, best_i]
        tabu_set.add(tuple(np.where(x_cur == 1)[0].tolist()))

        if f_cur > best_f + 1e-12:
            best_x = x_cur.copy()
            best_f = f_cur
            stall = 0
        else:
            stall += 1

    return best_x, best_f


# PMITS主循环

def run_pmits(A_hat, ld,
              pop_size=20,             # P：种群规模
              mutation_rate=50,        # M：突变百分比（0-100，论文默认 50%）
              ts_iters=50,             # I：TS 无改进停止迭代数
              max_generations=200,     # 最大代数
              time_limit=0,            # 时间限制（秒），0=不限制
              convergence_frac=0.5,    # 收敛比例阈值
              active_conv=True,        # 是否启用收敛停止准则
              verbose=True):

    lp = A_hat.shape[0]
    t_start = time.time()

    # 种群初始化
    X_star = []   # 每进程最优解
    X_prev = []   # 每进程上代局部最优

    for _ in range(pop_size):
        x1 = np.zeros(lp, dtype=np.int8)
        x1[np.random.choice(lp, ld, replace=False)] = 1
        x = x1
        x, _ = _tabu_search(x, A_hat, ld, ts_iters)
        X_star.append(x.copy())
        X_prev.append(x.copy())

    best_idx = np.argmax([0.5 * (x @ A_hat @ x) for x in X_star])
    global_best_x = X_star[best_idx].copy()
    global_best_f = 0.5 * (global_best_x @ A_hat @ global_best_x)
    history = [global_best_f]

    ts_map = {}          # 长期记忆：初始解键值 → (TS 结果, 目标值)
    last_improve = 0     # 最近改进代数

    if verbose:
        print(f"  PMITS init: P={pop_size}, M={mutation_rate}%, "
              f"I={ts_iters}, best_f={global_best_f:.4f}")

    # 主进化循环
    for gen in range(max_generations):
        # 时间限制检查
        if time_limit > 0 and (time.time() - t_start) > time_limit:
            if verbose:
                print(f"  PMITS: 达到时间限制 {time_limit}s，停止")
            break

        # 确定本代哪些个体接受突变
        n_mutate = max(1, int(pop_size * mutation_rate / 100.0))
        mutate_mask = np.zeros(pop_size, dtype=bool)
        mutate_mask[np.random.choice(pop_size, n_mutate, replace=False)] = True

        # 逐进程处理（串行模拟并行）
        for p in range(pop_size):
            kp = 1
            beta_p = min(0.9, (lp - 3) / lp)

            for _attempt in range(50):   # 安全上限
                # 偏斜交叉
                other = np.random.choice([i for i in range(pop_size) if i != p])
                lp = len(X_prev[p])
                mask = np.random.random(lp) < beta_p
                child = np.where(mask, X_prev[p], X_prev[other])

                excess = int(child.sum()) - ld
                if excess > 0:
                    ones = np.where(child == 1)[0]
                    child[np.random.choice(ones, excess, replace=False)] = 0
                elif excess < 0:
                    zeros = np.where(child == 0)[0]
                    child[np.random.choice(zeros, -excess, replace=False)] = 1
                beta_p = max(beta_p - 0.01, 0.3)

                # 突变
                if mutate_mask[p]:
                    xm = child.copy()
                    ones = np.where(xm == 1)[0]
                    zeros = np.where(xm == 0)[0]
                    if len(ones) == 0 or len(zeros) == 0:
                        return xm
                    n = min(kp, len(ones), len(zeros))
                    flip_ones = np.random.choice(ones, n, replace=False)
                    flip_zeros = np.random.choice(zeros, n, replace=False)
                    xm[flip_ones] = 0
                    xm[flip_zeros] = 1
                    child = xm
                    kp = min(kp + 1, lp // 2)

                # 局部搜索
                child = _local_search(child, A_hat, ld)
                child_key = tuple(np.where(child == 1)[0].tolist())

                # 接受准则
                if child_key == tuple(np.where(X_prev[p] == 1)[0].tolist()):
                    continue

                # 检索映射或执行
                if child_key in ts_map:
                    child, _ = ts_map[child_key]
                    child = child.copy()
                else:
                    child, child_f = _tabu_search(child, A_hat, ld, ts_iters)
                    ts_map[child_key] = (child.copy(), child_f)

                X_prev[p] = child.copy()

                # 更新进程最优
                child_f = 0.5 * (child @ A_hat @ child)
                if child_f > 0.5 * (X_star[p] @ A_hat @ X_star[p]) + 1e-12:
                    X_star[p] = child.copy()
                break

        # 全局最优更新
        all_f = [0.5 * (x @ A_hat @ x) for x in X_star]
        gen_best_idx = np.argmax(all_f)
        gen_best_f = all_f[gen_best_idx]

        if gen_best_f > global_best_f + 1e-12:
            global_best_x = X_star[gen_best_idx].copy()
            global_best_f = gen_best_f
            last_improve = gen

        history.append(global_best_f)

        # 收敛检查
        if active_conv:
            n_conv = sum(1 for x in X_star
                         if tuple(np.where(x == 1)[0].tolist()) == tuple(np.where(global_best_x == 1)[0].tolist()))
            if n_conv >= pop_size * convergence_frac:
                if verbose:
                    print(f"  PMITS converged: gen {gen+1}, "
                          f"{n_conv}/{pop_size}, "
                          f"t={time.time()-t_start:.1f}s")
                break

        # 反停滞重启
        if gen - last_improve > 1000:
            if verbose:
                print(f"  PMITS restart at gen {gen+1}")
            for p in range(pop_size):
                x2 = np.zeros(lp, dtype=np.int8)
                x2[np.random.choice(lp, ld, replace=False)] = 1
                x = x2
                x, _ = _tabu_search(x, A_hat, ld, ts_iters)
                X_prev[p] = x.copy()
                X_star[p] = x.copy()
            last_improve = gen

        # 进度日志
        if verbose and (gen + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  PMITS gen {gen+1}/{max_generations}, "
                  f"best_f={global_best_f:.4f}, t={elapsed:.0f}s")

    elapsed = time.time() - t_start
    if verbose:
        print(f"  PMITS done: best_f={global_best_f:.4f}, "
              f"gens={len(history)}, t={elapsed:.1f}s")

    return global_best_x, global_best_f, history


# 外部接口select_bits
def select_bits(A_hat, ld, **kwargs):
    return_history = kwargs.pop('return_history', False)
    best_x, best_f, history = run_pmits(A_hat, ld, **kwargs)
    selected = np.where(best_x == 1)[0]
    if return_history:
        return selected, best_f, history
    return selected, best_f
