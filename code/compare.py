"""多算法对比脚本，在不同哈希码长度下比较各算法的检索性能并绘图。
所有算法从同一个 288 位哈希池中选择比特。
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils import (compute_affinity_matrix, compute_all_metrics,
                    objective as obj_func)
from hashing.hash_pool import load_hash_pool
from algorithm.ga_al import select_bits as ga_select
from algorithm.random_al import select_bits as random_select
from algorithm.pmits_al import select_bits as pmits_select
from algorithm.iga_al import select_bits as iga_select
from algorithm.pso_adv_al import select_bits as pso_select

# 待对比的算法列表
# 每项格式: (算法名称, select_bits 函数, 关键字参数字典)

ALGOS = [
    ('Random', random_select, {}),
    ('GA',     ga_select,     {}),
    ('PMITS',  pmits_select,  {'pop_size': 10, 'mutation_rate': 50,
                                'ts_iters': 30, 'max_generations': 50,
                                'time_limit': 0, 'active_conv': True}),
    ('IGA',    iga_select,    {}),
    ('BPSO',   pso_select,    {}),
]

# 各算法的最大代数预算（用于收敛图中虚线延展到右边界）
GEN_BUDGETS = {'GA': 200, 'PMITS': 50, 'IGA': 60, 'BPSO': 100}

# 优化选择 + 检索评估

def run_experiment(algo_name, algo_fn, algo_kwargs, A_hat, ld, pool_test,
                   test_labels, verbose=True):
    # 从 288 位哈希池中优化选出 ld 个比特，并评估检索性能。

    t0 = time.time()
    if verbose:
        print(f"\n{'='*50}")
        print(f"Running {algo_name}, LD={ld}")
        print(f"{'='*50}")

    # 优化选择（含收敛历史）
    hist_kwargs = {**algo_kwargs, 'return_history': True}
    result_tuple = algo_fn(A_hat, ld, **hist_kwargs)
    selected_idx, best_f = result_tuple[0], result_tuple[1]
    conv_history = result_tuple[2] if len(result_tuple) > 2 else []
    elapsed = time.time() - t0
    if verbose:
        print(f"  Selected {len(selected_idx)} bits, objective={best_f:.4f}, "
              f"time={elapsed:.1f}s")
        sorted_idx = np.sort(selected_idx)
        idx_str = ', '.join(str(i) for i in sorted_idx)
        print(f"  Selected bits: [{idx_str}]")

    # 构建检索码
    db_codes = pool_test[selected_idx].T     # (n_test, ld)
    n_q = min(NUM_QUERY, db_codes.shape[0] // 2)
    query_codes = db_codes[:n_q]              # 前 n_q 个作为查询
    db_codes_sub = db_codes[n_q:]             # 其余作为检索库
    query_labels = test_labels[:n_q]
    db_labels = test_labels[n_q:]

    # 计算检索指标（一次汉明距离，四个指标）
    nk = min(TOP_NK, len(db_labels))
    map_val, prec, rec, ph2 = compute_all_metrics(
        query_codes, db_codes_sub, query_labels, db_labels, top_k=nk)

    if verbose:
        print(f"  MAP@{nk}={map_val*100:.2f}%, P@{nk}={prec*100:.2f}%, "
              f"R@{nk}={rec*100:.2f}%, PH2={ph2*100:.2f}%")

    return {'objective': best_f, 'MAP': map_val, 'precision': prec,
            'recall': rec, 'PH2': ph2, 'selected_idx': selected_idx,
            'time': elapsed, 'history': conv_history}

# 绘图生成五指标 × 多算法的对比折线图

def plot_results(all_results, ld_list):
    """为每种算法的每个指标绘制随 LD 变化的折线图。

    生成 2×3 子图布局（第 6 格留空），分别展示：
        Objective, MAP, Precision@K, Recall@K, PH2
    """
    algo_names = [a[0] for a in ALGOS]
    colors = ['#E63946', '#3B7DD8', '#2ECC40', '#E67E22', '#8E44AD', '#16A085',
              '#F39C12', '#E91E90'][:len(algo_names)]
    markers = ['o', 's', '^', 'D', 'v', 'p']                    # 不同算法用不同标记

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    metrics = ['objective', 'MAP', 'precision', 'recall']
    titles = ['Objective Value', 'MAP', f'Precision@{TOP_NK}',
              f'Recall@{TOP_NK}']

    for ax, metric, title in zip(axes.flat[:4], metrics, titles):
        for i, name in enumerate(algo_names):
            if metric == 'objective' and name == 'Random':
                continue
            values = [all_results[name][ld][metric] for ld in ld_list]
            if metric in ('MAP', 'precision', 'recall'):
                values = [v * 100 for v in values]

            kw = dict(marker=markers[i % len(markers)], color=colors[i],
                      linewidth=2, markersize=8, label=name)

            ax.plot(ld_list, values, linestyle='-', **kw)
        ax.set_xlabel('Hash Code Length (ld)', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(POOL_DIR, 'comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"指标对比图已保存到 {save_path}")
    plt.close()


def plot_time_boxplot(all_raw_times, ld_list):
    """生成各 LD 下四种智能算法的耗时箱线图。"""
    smart_algos = ['GA', 'PMITS', 'IGA', 'BPSO']
    colors = ['#3B7DD8', '#2ECC40', '#E67E22', '#8E44AD']
    n = len(ld_list)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, ld in enumerate(ld_list):
        ax = axes[idx]
        data = [all_raw_times[name][ld] for name in smart_algos]
        bp = ax.boxplot(data, labels=smart_algos, patch_artist=True,
                        widths=0.5, showfliers=False,
                        medianprops={'color': 'black', 'linewidth': 1.5})
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(f'LD={ld}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Time (s)', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', labelsize=8)

    for idx in range(len(ld_list), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(POOL_DIR, 'comparison_time_boxplot.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"耗时箱线图已保存到 {save_path}")
    plt.close()


def plot_convergence(all_histories, ld_list):
    """生成各 LD 下四种智能算法的收敛曲线。早停部分以虚线延展。"""
    smart_algos = ['GA', 'PMITS', 'IGA', 'BPSO']
    colors = ['#3B7DD8', '#2ECC40', '#E67E22', '#8E44AD']
    n = len(ld_list)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, ld in enumerate(ld_list):
        ax = axes[idx]
        for i, name in enumerate(smart_algos):
            histories = all_histories[name][ld]
            if not histories or len(histories[0]) == 0:
                continue

            lengths = [len(h) for h in histories]
            is_pmits = (name == 'PMITS')

            # 将所有 history 补齐到 200 代
            padded = np.array([
                np.pad(h, (0, 200 - len(h)), 'edge')
                if len(h) < 200 else h[:200]
                for h in histories
            ])
            mean_curve = np.mean(padded, axis=0)
            std_curve = np.std(padded, axis=0)
            gens = np.arange(1, 201)

            # 中位数停止点为实线→虚线分界
            med_stop = int(np.median(lengths))
            lw_solid = 3 if is_pmits else 1.5
            lw_dash = 2.5 if is_pmits else 1.2
            alpha_fill = 0.2 if is_pmits else 0.08

            ax.plot(gens[:med_stop], mean_curve[:med_stop],
                    color=colors[i], linewidth=lw_solid, label=name,
                    zorder=5 if is_pmits else 3)
            ax.fill_between(gens[:med_stop],
                            mean_curve[:med_stop] - std_curve[:med_stop],
                            mean_curve[:med_stop] + std_curve[:med_stop],
                            color=colors[i], alpha=alpha_fill)
            if med_stop < 200:
                ax.plot(gens[med_stop - 1:], mean_curve[med_stop - 1:],
                        color=colors[i], linewidth=lw_dash, linestyle='--',
                        alpha=0.9 if is_pmits else 0.7,
                        zorder=5 if is_pmits else 3)

        ax.set_title(f'LD={ld}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Generation', fontsize=9)
        ax.set_ylabel('Objective', fontsize=9)
        ax.set_xlim(1, 200)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)

    for idx in range(len(ld_list), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(POOL_DIR, 'comparison_convergence.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"收敛曲线图已保存到 {save_path}")
    plt.close()


def plot_mean_time(all_raw_times, ld_list):
    """四种智能算法全部码长的耗时汇总箱线图（每算法一个箱线）。"""
    smart_algos = ['GA', 'PMITS', 'IGA', 'BPSO']
    colors = ['#3B7DD8', '#2ECC40', '#E67E22', '#8E44AD']

    fig, ax = plt.subplots(figsize=(8, 6))

    data = []
    for name in smart_algos:
        all_times = []
        for ld in ld_list:
            all_times.extend(all_raw_times[name][ld])
        data.append(all_times)

    bp = ax.boxplot(data, labels=smart_algos, patch_artist=True,
                    widths=0.45, showfliers=False,
                    medianprops={'color': 'black', 'linewidth': 1.5})

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # 标注统计信息
    for i, (name, d) in enumerate(zip(smart_algos, data)):
        arr = np.array(d)
        median = np.median(arr)
        mean = np.mean(arr)
        q3 = np.percentile(arr, 75)
        ax.text(i + 1, q3 + 0.05, f'median={median:.5f}s\nmean={mean:.5f}s',
                ha='center', fontsize=8, fontweight='bold', color=colors[i])

    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title('Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(POOL_DIR, 'comparison_mean_time.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"耗时汇总箱线图已保存到 {save_path}")
    plt.close()

# 主函数

def main():
    """主流水线：加载池 → 亲和度矩阵 → 遍历算法×码长 → 汇总 → 绘图。"""
    # 强制行缓冲，确保后台运行时进度实时可见
    sys.stdout.reconfigure(line_buffering=True)

    NUM_RUNS = 50   # 多次实验取均值

    # 用种子 42 生成 50 个新种子
    seed_rng = np.random.RandomState(SEED)
    run_seeds = seed_rng.randint(0, 2**31-1, NUM_RUNS)

    print("算法对比")
    print(f"基准种子: {SEED}")
    print(f"实验次数: {NUM_RUNS} 次取均值")
    print(f"对比算法: {[a[0] for a in ALGOS]}")
    print(f"哈希码长度: {LD_LIST}")

    # 加载哈希池
    t0 = time.time()
    pool_train, pool_test, train_feat, test_feat, train_labels, test_labels = \
        load_hash_pool()
    print(f"哈希池加载完成 ({time.time()-t0:.1f}s): "
          f"lp={pool_train.shape[0]}, n_train={pool_train.shape[1]}")

    # 亲和度矩阵
    import hashlib, json
    cache_path = os.path.join(POOL_DIR, 'A_hat.npz')
    # 用池形状 + 配置参数做缓存指纹
    cache_key = hashlib.md5(json.dumps({
        'lp': int(pool_train.shape[0]), 'n': int(pool_train.shape[1]),
        'k': K_NEIGHBORS, 'xi': XI, 'beta': BETA,
    }, sort_keys=True).encode()).hexdigest()

    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached.get('key', '') == cache_key:
            A_hat = cached['A_hat']
            print(f"\n亲和度矩阵从缓存加载 (形状={A_hat.shape})")
        else:
            print("\n缓存指纹不匹配，重新计算亲和度矩阵...")
            t0 = time.time()
            A_hat, phi, A_mat = compute_affinity_matrix(
                pool_train, features=train_feat, k=K_NEIGHBORS, xi=XI, beta=BETA)
            np.savez_compressed(cache_path, A_hat=A_hat, key=cache_key)
            print(f"计算+缓存完成 ({time.time()-t0:.1f}s)")
    else:
        t0 = time.time()
        print("\n计算亲和度矩阵（首次，将缓存到 pool/A_hat.npz）")
        A_hat, phi, A_mat = compute_affinity_matrix(
            pool_train, features=train_feat, k=K_NEIGHBORS, xi=XI, beta=BETA)
        np.savez_compressed(cache_path, A_hat=A_hat, key=cache_key)
        print(f"计算+缓存完成 ({time.time()-t0:.1f}s)")

    # 遍历算法 × 码长 × 50 次重复，取均值
    # 初始化累加器
    all_results = {}
    all_raw_times = {}      # 用于箱线图
    all_histories = {}      # 用于收敛图
    for name, _, _ in ALGOS:
        all_results[name] = {}
        all_raw_times[name] = {}
        all_histories[name] = {}
        for ld in LD_LIST:
            all_results[name][ld] = {
                'objective': [], 'MAP': [], 'precision': [],
                'recall': [], 'PH2': [], 'time': [],
            }
            all_raw_times[name][ld] = []
            all_histories[name][ld] = []

    total_exp = len(ALGOS) * len(LD_LIST) * NUM_RUNS
    done = 0
    print(f"\n运行实验（共 {total_exp} 次）")

    for name, algo_fn, kwargs in ALGOS:
        print(f"\n{'='*50}\n[{name}]\n{'='*50}")
        for ld in LD_LIST:
            acc = all_results[name][ld]
            for run_i, seed in enumerate(run_seeds):
                np.random.seed(seed)
                # 压制单次运行的详细输出，仅保留进度
                quiet_kwargs = {**kwargs, 'verbose': False}
                result = run_experiment(name, algo_fn, quiet_kwargs, A_hat, ld,
                                        pool_test, test_labels, verbose=False)
                for key in acc:
                    acc[key].append(result[key])
                all_raw_times[name][ld].append(result['time'])
                all_histories[name][ld].append(result['history'])
                done += 1
                if (done % 50) == 0:
                    print(f"  进度: {done}/{total_exp}")

    # 计算均值并存储 selected_idx（取最后一次的，索引本身不平均）
    for name, _, _ in ALGOS:
        for ld in LD_LIST:
            acc = all_results[name][ld]
            mean_result = {key: np.mean(vals) for key, vals in acc.items()
                           if key != 'selected_idx'}
            mean_result['selected_idx'] = result['selected_idx']  # 保留最后次的
            all_results[name][ld] = mean_result

    # 阶段 4：打印汇总表
    print("\n\n汇总表")
    header = (f"{'Algorithm':<10} {'LD':<5} {'Objective':<12} "
              f"{'MAP(%)':<10} {'P(%)':<10} {'R(%)':<10} {'PH2(%)':<10}")
    print(header)
    print('-' * len(header))
    for name, _, _ in ALGOS:
        for ld in LD_LIST:
            r = all_results[name][ld]
            print(f"{name:<10} {ld:<5} {r['objective']:<12.4f} "
                  f"{r['MAP']*100:<10.2f} {r['precision']*100:<10.2f} "
                  f"{r['recall']*100:<10.2f} {r['PH2']*100:<10.2f}")

    # 绘图
    print("\n生成对比图")
    plot_results(all_results, LD_LIST)
    plot_time_boxplot(all_raw_times, LD_LIST)
    plot_convergence(all_histories, LD_LIST)
    plot_mean_time(all_raw_times, LD_LIST)

    # 保存完整结果
    np.savez(os.path.join(POOL_DIR, 'comparison_results.npz'),
             results=all_results, ld_list=LD_LIST)

    print("全部完成。")


if __name__ == '__main__':
    main()
