"""改进型遗传算法IGA
相对于基本遗传算法的改进：
  1. 禁忌搜索局部寻优，系统性地探索单步交换邻域（向量化实现）
  2. 路径重连重组，精英解之间的结构化组合
  3. 确定性拥挤，采用小生境技术维持种群多样性
"""
import numpy as np


# 向量化禁忌搜索

def _tabu_search(x_init, A_hat, ld, max_iter=300, max_stall=15):
    # 向量化禁忌搜索，带停滞早停机制。
    lp = A_hat.shape[0]
    x_cur = x_init.copy()
    f_cur = 0.5 * float(x_cur @ A_hat @ x_cur)
    x_best, f_best = x_cur.copy(), f_cur

    Ax = A_hat @ x_cur.astype(np.float64)
    diag = np.diag(A_hat).copy()

    # 禁忌表：字典映射 (j,i) -> 过期迭代次数
    tabu_tenure_min, tabu_tenure_max = 8, 25
    tabu = {}

    stall = 0
    it = 0
    while it < max_iter and stall < max_stall:
        ones = np.where(x_cur == 1)[0]
        zeros = np.where(x_cur == 0)[0]
        n1, n0 = len(ones), len(zeros)

        # 向量化增量计算
        delta = (2.0 * (Ax[zeros][:, None] - Ax[ones][None, :]) +
                 diag[zeros][:, None] + diag[ones][None, :] -
                 2.0 * A_hat[zeros[:, None], ones[None, :]])

        # 按delta降序排列；扫描寻找最佳的非禁忌移动
        flat_idx = np.argsort(delta.ravel())[::-1]

        best_move = None
        best_delta_val = -float('inf')

        for fidx in flat_idx:
            zi = fidx // n1  # zeros中的索引
            oi = fidx % n1   # ones中的索引
            delta_val = delta[zi, oi]
            i, j = ones[oi], zeros[zi]  # i: 1->0, j: 0->1
            f_new = f_cur + 0.5 * delta_val

            # 渴望准则：如果优于全局最优，立即接受
            if f_new > f_best:
                best_move = (i, j)
                best_delta_val = delta_val
                break

            # 检查禁忌
            is_tabu = tabu.get((j, i), -1) >= it
            if not is_tabu and delta_val > best_delta_val:
                best_delta_val = delta_val
                best_move = (i, j)

        if best_move is None:
            break  # 所有移动均被禁忌或已穷尽

        # 执行移动
        i, j = best_move
        x_cur[i], x_cur[j] = 0, 1
        f_cur += 0.5 * best_delta_val
        Ax = Ax - A_hat[:, i] + A_hat[:, j]

        # 将逆向移动设为禁忌
        tenure = np.random.randint(tabu_tenure_min, tabu_tenure_max + 1)
        tabu[(j, i)] = it + tenure

        # 定期清理
        if it % 60 == 0:
            tabu = {k: v for k, v in tabu.items() if v >= it}

        if f_cur > f_best:
            x_best, f_best = x_cur.copy(), f_cur
            stall = 0
        else:
            stall += 1

        it += 1

    return x_best, f_best

# 向量化路径重连

def _path_relinking(x_a, x_b, A_hat, ld, max_stall=15):
    """贪婪路径重连：探索从 x_a 到 x_b 的路径。
    每一步通过向量化增量计算评估所有剩余交换，然后贪婪地选择最佳的一个。当连续max_stall步未找到更优解时提前终止。返回路径上的最优解。
    """
    # 快速检查：如果太相似，跳过 PR
    diff_bits = np.sum(x_a != x_b)
    if diff_bits <= 2:
        f_a = 0.5 * float(x_a @ A_hat @ x_a)
        return x_a.copy(), f_a

    x_cur = x_a.copy()
    f_cur = 0.5 * float(x_cur @ A_hat @ x_cur)
    x_best, f_best = x_cur.copy(), f_cur

    Ax = A_hat @ x_cur.astype(np.float64)
    diag = np.diag(A_hat)

    otr = np.where((x_cur == 1) & (x_b == 0))[0]        # 需要移除的 1
    zta = np.where((x_cur == 0) & (x_b == 1))[0]        # 需要添加的 0

    if len(otr) == 0 or len(zta) == 0:
        return x_best, f_best

    stall = 0
    max_steps = min(len(otr), 30)  # 路径长度上限为 30 步

    for step in range(max_steps):
        n_rem = len(otr)
        n_add = len(zta)
        if n_rem == 0 or n_add == 0:
            break

        # 所有剩余交换对的向量化增量计算
        delta = (2.0 * (Ax[zta][:, None] - Ax[otr][None, :]) +
                 diag[zta][:, None] + diag[otr][None, :] -
                 2.0 * A_hat[zta[:, None], otr[None, :]])  # (n_add, n_rem)

        if delta.size == 0:
            break

        best_flat = np.argmax(delta.ravel())
        zi, oi = best_flat // n_rem, best_flat % n_rem
        best_delta = delta[zi, oi]

        i, j = otr[oi], zta[zi]
        x_cur[i], x_cur[j] = 0, 1
        f_cur += 0.5 * best_delta
        Ax = Ax - A_hat[:, i] + A_hat[:, j]

        # 通过布尔掩码移除已使用的位置
        otr = np.delete(otr, oi)
        zta = np.delete(zta, zi)

        if f_cur > f_best:
            x_best, f_best = x_cur.copy(), f_cur
            stall = 0
        else:
            stall += 1
            if stall >= max_stall:
                break

    return x_best, f_best

# 基本算子

def _hamming(x1, x2):
    return np.sum(x1 != x2)

# IGA主循环（文化基因算法）
def run_iga(A_hat, ld, pop_size=60, generations=60,
            elite_pool_size=10, ts_iter=40, deep_ts_iter=80,
            patience=25, verbose=True):

    lp = A_hat.shape[0]
    if pop_size % 2 != 0:
        pop_size += 1

    # 随机初始化

    x_temp = np.zeros(lp, dtype=np.int8)
    x_temp[np.random.choice(lp, ld, replace=False)] = 1
    pop = np.array([x_temp for _ in range(pop_size)])
    fit = np.array([0.5 * float(x @ A_hat @ x) for x in pop])

    best_idx = np.argmax(fit)
    best_x, best_f = pop[best_idx].copy(), fit[best_idx]
    history = [best_f]

    elite_pool = [best_x.copy()]
    elite_fits = [best_f]
    stagnation = 0
    div = 999.0          # 初始假设种群多样
    last_ts_gen = -999   # 记录上次调用禁忌搜索的代数

    for gen in range(generations):
        # 1. 自适应禁忌搜索
        ts_interval = 1 if div > 1.0 else 5
        ts_iters = ts_iter if div > 1.0 else max(15, ts_iter // 2)
        ts_stall = 10 if div > 1.0 else 8

        if gen - last_ts_gen >= ts_interval:
            top_idx = np.argmax(fit)
            x_ts, f_ts = _tabu_search(pop[top_idx], A_hat, ld, max_iter=ts_iters, max_stall=ts_stall)
            if f_ts > fit[top_idx]:
                pop[top_idx] = x_ts
                fit[top_idx] = f_ts
                if f_ts > best_f:
                    best_x, best_f = x_ts.copy(), f_ts
                    stagnation = 0
            last_ts_gen = gen

        # 2. 更新精英池
        sorted_idx = np.argsort(fit)[::-1]
        for idx in sorted_idx:
            candidate = pop[idx]
            candidate_f = fit[idx]
            # 仅当与现有精英足够不同时才添加
            min_dist = min([_hamming(candidate, e) for e in elite_pool]) if elite_pool else float('inf')
            if min_dist > 2 or candidate_f > max(elite_fits) + 1e-8:
                elite_pool.append(candidate.copy())
                elite_fits.append(candidate_f)
                if len(elite_pool) > elite_pool_size:
                    worst = np.argmin(elite_fits)
                    elite_pool.pop(worst)
                    elite_fits.pop(worst)
                if len(elite_pool) >= elite_pool_size:
                    break

        # 3. 构建下一代
        next_pop, next_fit = [], []

        # 精英保留：保留前 2 个
        elite2_idx = np.argsort(fit)[-2:]
        for idx in elite2_idx:
            next_pop.append(pop[idx].copy())
            next_fit.append(fit[idx])

        while len(next_pop) < pop_size:
            # 锦标赛选择
            candidates1 = np.random.choice(len(fit), 3, replace=False)
            p1_idx = candidates1[np.argmax(fit[candidates1])]
            candidates2 = np.random.choice(len(fit), 3, replace=False)
            p2_idx = candidates2[np.argmax(fit[candidates2])]
            p1, p2 = pop[p1_idx], pop[p2_idx]

            # 重组：路径重连（0.3 概率）或交叉（0.7 概率）
            if np.random.random() < 0.3 and len(elite_pool) >= 2:
                # PR：子代沿着从亲代到随机精英的路径前进
                e1 = elite_pool[np.random.choice(len(elite_pool))]
                c1, f_c1 = _path_relinking(p1, e1, A_hat, ld)
                e2 = elite_pool[np.random.choice(len(elite_pool))]
                c2, f_c2 = _path_relinking(p2, e2, A_hat, ld)
            else:
                # 标准交叉 + 变异
                lp = len(p1)
                mask = np.random.random(lp) < 0.5
                c1 = np.where(mask, p1, p2)
                c2 = np.where(mask, p2, p1)
                for c in (c1, c2):
                    diff = int(c.sum()) - ld
                    if diff > 0:
                        ones = np.where(c == 1)[0]
                        c[np.random.choice(ones, diff, replace=False)] = 0
                    elif diff < 0:
                        zeros = np.where(c == 0)[0]
                        c[np.random.choice(zeros, -diff, replace=False)] = 1
                if np.random.random() > 0.08:
                    c1 = c1.copy()
                else:
                    xm1 = c1.copy()
                    ones = np.where(xm1 == 1)[0]
                    zeros = np.where(xm1 == 0)[0]
                    if len(ones) > 0 and len(zeros) > 0:
                        i, j = np.random.choice(ones), np.random.choice(zeros)
                        xm1[i], xm1[j] = 0, 1
                    c1 = xm1
                if np.random.random() > 0.08:
                    c2 = c2.copy()
                else:
                    xm2 = c2.copy()
                    ones = np.where(xm2 == 1)[0]
                    zeros = np.where(xm2 == 0)[0]
                    if len(ones) > 0 and len(zeros) > 0:
                        i, j = np.random.choice(ones), np.random.choice(zeros)
                        xm2[i], xm2[j] = 0, 1
                    c2 = xm2
                f_c1 = 0.5 * float(c1 @ A_hat @ c1)
                f_c2 = 0.5 * float(c2 @ A_hat @ c2)

            f_p1, f_p2 = fit[p1_idx], fit[p2_idx]

            # 确定性拥挤：子代与最相似的亲代竞争
            if _hamming(c1, p1) <= _hamming(c1, p2):
                w1 = c1.copy() if f_c1 > f_p1 else p1.copy()
            else:
                w1 = c1.copy() if f_c1 > f_p2 else p2.copy()
            if _hamming(c2, p1) <= _hamming(c2, p2):
                w2 = c2.copy() if f_c2 > f_p1 else p1.copy()
            else:
                w2 = c2.copy() if f_c2 > f_p2 else p2.copy()

            next_pop.append(w1)
            next_fit.append(0.5 * float(w1 @ A_hat @ w1))
            if len(next_pop) < pop_size:
                next_pop.append(w2)
                next_fit.append(0.5 * float(w2 @ A_hat @ w2))

        pop = np.array(next_pop[:pop_size])
        fit = np.array(next_fit[:pop_size])

        # 更新最优解并跟踪多样性
        cur_best_idx = np.argmax(fit)
        if fit[cur_best_idx] > best_f:
            best_x, best_f = pop[cur_best_idx].copy(), fit[cur_best_idx]
            stagnation = 0
        else:
            stagnation += 1

        history.append(best_f)

        # 计算种群多样性（平均成对海明距离）
        div = np.mean([_hamming(pop[i], pop[j])
                       for i in range(min(10, pop_size))
                       for j in range(i + 1, min(10, pop_size))])

        # 停滞时深度禁忌搜索（上限为 deep_ts_iter）
        if stagnation == 15:
            x_deep, f_deep = _tabu_search(best_x, A_hat, ld,
                                          max_iter=deep_ts_iter, max_stall=15)
            if f_deep > best_f:
                best_x, best_f = x_deep, f_deep
                stagnation = 0
                if verbose:
                    print(f"  IGA gen {gen+1:3d}: 深度禁忌搜索突破 -> best_f={best_f:.4f}")

        # 6. 提前终止
        if stagnation >= patience:
            if verbose:
                print(f"  IGA 提前终止于 gen {gen+1}: "
                      f"连续 {patience} 代无改进，best_f={best_f:.4f}")
            break

        if verbose and (gen + 1) % 25 == 0:
            ts_mode = "快速" if div > 1.0 else "稀疏"
            print(f"  IGA gen {gen+1:3d}/{generations} | best_f={best_f:.4f} | "
                  f"div={div:.1f} | ts={ts_mode} | stag={stagnation:2d}")

    # 最终禁忌搜索（停滞早停，上限 80 次迭代）
    x_final, f_final = _tabu_search(best_x, A_hat, ld, max_iter=80, max_stall=15)
    if f_final > best_f:
        best_x, best_f = x_final, f_final
        history.append(best_f)
        if verbose:
            print(f"  IGA 最终禁忌搜索: {f_final:.4f}")

    return best_x, best_f, history

# 外部接口

def select_bits(A_hat, ld, **kwargs):
    """选择 ld 个比特。返回 (selected_indices, objective_value, [history])。"""
    return_history = kwargs.pop('return_history', False)
    best_x, best_f, history = run_iga(A_hat, ld, **kwargs)
    selected = np.where(best_x == 1)[0]
    if return_history:
        return selected, best_f, history
    return selected, best_f