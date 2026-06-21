"""PMITS 选择 vs 四种经典哈希方法 对比脚本。

对比对象：
    PMITS       —— 从256位传统哈希池中优化选出 ld 个比特
    KLSH        —— 仅使用 KLSH 方法的前 ld 位（比特 0-71）
    PCAH        —— 仅使用 PCAH 方法的前 ld 位（比特 72-143）
    LSH         —— 仅使用 LSH 方法的前 ld 位（比特 144-215）
    回答"PMITS 的优化选择在经典哈希方法上的表现如何"。

"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from utils import (compute_affinity_matrix, compute_all_metrics)
from algorithm.pmits_al import select_bits as pmits_select

# 哈希方法定义 —— 每种方法在池中的比特范围

HASH_METHODS_2 = {
    'KLSH': (0, BITS_PER_METHOD),                          # 比特   0–71
    'PCAH': (BITS_PER_METHOD, BITS_PER_METHOD * 2),        # 比特  72–143
    'LSH':  (BITS_PER_METHOD * 2, BITS_PER_METHOD * 3),    # 比特 144–215
}

# 传统哈希池文件路径
TRADITION_POOL_PATH = os.path.join(POOL_DIR, 'tradition_hash_pool.npz')


def load_tradition_hash_pool():
    # 加载传统哈希池文件。

    if not os.path.exists(TRADITION_POOL_PATH):
        raise FileNotFoundError(
            f"传统哈希池文件不存在: {TRADITION_POOL_PATH}\n"
            f"请先运行 'python hashing/hash_pool_2.py' 构建池。")
    data = np.load(TRADITION_POOL_PATH)
    return (data['pool_train'].astype(np.int8), data['pool_test'].astype(np.int8),
            data['train_feat'], data['test_feat'],
            data['train_labels'], data['test_labels'])


def evaluate_codes(test_codes, test_labels):
    """给定哈希码和标签，计算检索指标。"""
    n_q = min(NUM_QUERY, test_codes.shape[0] // 2)
    query_codes = test_codes[:n_q]
    db_codes = test_codes[n_q:]
    query_labels = test_labels[:n_q]
    db_labels = test_labels[n_q:]

    nk = min(TOP_NK, len(db_labels))
    map_val, prec, rec, ph2 = compute_all_metrics(
        query_codes, db_codes, query_labels, db_labels, top_k=nk)
    return map_val, prec, rec, ph2


# 单次方法评估

def eval_pmits(A_hat, ld, pool_test, test_labels, verbose=True):
    """PMITS 选择 + 评估。"""
    t0 = time.time()
    idx, obj = pmits_select(A_hat, ld, pop_size=10, mutation_rate=50,
                            ts_iters=30, max_generations=50, verbose=verbose)
    codes = pool_test[idx].T.astype(np.int8)
    elapsed = time.time() - t0
    map_val, prec, rec, ph2 = evaluate_codes(codes, test_labels)
    if verbose:
        print(f"  PMITS LD={ld:>3}: MAP={map_val*100:.2f}%, "
              f"P@900={prec*100:.2f}%, time={elapsed:.1f}s")
    return {'MAP': map_val, 'precision': prec, 'recall': rec, 'PH2': ph2,
            'objective': obj, 'selected_idx': idx, 'time': elapsed}


def eval_method(method_name, ld, pool_test, test_labels, verbose=True):
    """单一哈希方法评估,取该方法的前 ld 个比特。"""
    t0 = time.time()
    start, end = HASH_METHODS_2[method_name]
    idx = np.arange(start, start + ld)
    codes = pool_test[idx].T.astype(np.int8)
    elapsed = time.time() - t0
    map_val, prec, rec, ph2 = evaluate_codes(codes, test_labels)
    if verbose:
        print(f"  {method_name:<5s} LD={ld:>3}: MAP={map_val*100:.2f}%, "
              f"P@900={prec*100:.2f}%, time={elapsed:.3f}s")
    return {'MAP': map_val, 'precision': prec, 'recall': rec, 'PH2': ph2,
            'objective': 0.0, 'selected_idx': idx, 'time': elapsed}

# 绘图

def plot_comparison(results, ld_list):
    """生成 PMITS vs 四种经典哈希方法的对比折线图。"""
    method_names = ['PMITS'] + list(HASH_METHODS_2.keys())
    colors = ['#E63946', '#3B7DD8', '#2ECC40', '#E67E22']
    markers = ['D', 'o', 's', '^']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = ['MAP', 'precision', 'recall', 'PH2']
    titles = ['MAP@900', 'Precision@900', 'Recall@900', 'PH2']

    for ax, metric, title in zip(axes.flat[:4], metrics, titles):
        for i, name in enumerate(method_names):
            values = [results[name][ld][metric] * 100 for ld in ld_list]

            kw = dict(marker=markers[i], color=colors[i],
                      linewidth=2.5 if name == 'PMITS' else 1.8,
                      markersize=9 if name == 'PMITS' else 7,
                      alpha=0.95, label=name)

            if metric == 'PH2':
                split = next((j for j, v in enumerate(values) if v == 0), None)
                if split is not None and split > 0:
                    kw_s = {**kw, 'label': name}
                    ax.plot(ld_list[:split], values[:split], linestyle='-', **kw_s)
                    kw_d = {**kw, 'markerfacecolor': 'white', 'label': ''}
                    ax.plot(ld_list[split-1:], values[split-1:],
                            linestyle='--', **kw_d)
                    continue
                elif split is not None:
                    ax.plot(ld_list, values, linestyle='--', **kw)
                    continue

            ax.plot(ld_list, values, linestyle='-', **kw)
        ax.set_xlabel('Hash Code Length (ld)', fontsize=11)
        ax.set_ylabel(f'{title} (%)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(POOL_DIR, 'comparison_methods_2.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"折线对比图已保存到 {save_path}")
    plt.close()


def plot_bar_comparison(results, ld_list):
    """生成各 LD 下 PMITS vs 四种经典哈希方法的 MAP 柱状对比图。"""
    method_names = ['PMITS'] + list(HASH_METHODS_2.keys())
    colors = ['#E63946', '#3B7DD8', '#2ECC40', '#E67E22']
    n = len(ld_list)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, ld in enumerate(ld_list):
        ax = axes[idx]
        map_values = [results[name][ld]['MAP'] * 100 for name in method_names]
        bars = ax.bar(method_names, map_values, color=colors, edgecolor='white',
                      linewidth=1.2)
        for bar, val in zip(bars, map_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7,
                    fontweight='bold')
        ax.set_title(f'LD={ld}', fontsize=12, fontweight='bold')
        ax.set_ylabel('MAP@900 (%)', fontsize=9)
        ax.set_ylim(15, max(map_values) * 1.08 if max(map_values) > 0 else 1)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=15, labelsize=8)

    for idx in range(len(ld_list), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(POOL_DIR, 'comparison_methods_2_bars.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"柱状对比图已保存到 {save_path}")
    plt.close()

# 主函数

def main():
    sys.stdout.reconfigure(line_buffering=True)
    NUM_RUNS = 50

    seed_rng = np.random.RandomState(SEED)
    run_seeds = seed_rng.randint(0, 2**31-1, NUM_RUNS)

    print("=" * 60)
    print("PMITS 选择 vs 四种经典哈希方法 对比")
    print("=" * 60)
    print(f"基准种子: {SEED}, 实验次数: {NUM_RUNS} 次取均值")
    print(f"对比方法: PMITS + {list(HASH_METHODS_2.keys())}")
    print(f"哈希码长度: {LD_LIST}")
    print(f"池结构: KLSH(0-71) PCAH(72-143) LSH(144-215)")

    # 加载传统哈希池
    pool_train, pool_test, train_feat, test_feat, train_labels, test_labels = \
        load_tradition_hash_pool()

    # 亲和度矩阵
    import hashlib, json
    cache_path = os.path.join(POOL_DIR, 'A_hat_tradition.npz')
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
            print("\n缓存指纹不匹配，重新计算...")
            t0 = time.time()
            A_hat, phi, A_mat = compute_affinity_matrix(
                pool_train, features=train_feat, k=K_NEIGHBORS, xi=XI, beta=BETA)
            np.savez_compressed(cache_path, A_hat=A_hat, key=cache_key)
            print(f"完成 ({time.time()-t0:.1f}s)")
    else:
        print("\n计算亲和度矩阵")
        t0 = time.time()
        A_hat, phi, A_mat = compute_affinity_matrix(
            pool_train, features=train_feat, k=K_NEIGHBORS, xi=XI, beta=BETA)
        np.savez_compressed(cache_path, A_hat=A_hat, key=cache_key)
        print(f"完成 ({time.time()-t0:.1f}s)")

    #  四种经典哈希方法
    results = {
        'PMITS': {},
        'KLSH': {}, 'PCAH': {}, 'LSH': {},
    }
    print("\n评估经典哈希方法（一次性）")
    for ld in LD_LIST:
        for method_name in HASH_METHODS_2:
            results[method_name][ld] = eval_method(method_name, ld, pool_test,
                                                    test_labels, verbose=False)
        print(f"  LD={ld}: KLSH/ PCAH/ LSH 完成")

    # PMITS（50 次取均值）
    print(f"\nPMITS {NUM_RUNS} 次实验取均值")
    for ld in LD_LIST:
        acc = {'MAP': [], 'precision': [], 'recall': [], 'PH2': [],
               'objective': [], 'time': []}
        for run_i, seed in enumerate(run_seeds):
            np.random.seed(seed)
            r = eval_pmits(A_hat, ld, pool_test, test_labels, verbose=False)
            for key in acc:
                acc[key].append(r[key])
            if (run_i + 1) % 10 == 0:
                print(f"  LD={ld}: {run_i+1}/{NUM_RUNS}")
        mean_r = {key: np.mean(vals) for key, vals in acc.items()}
        mean_r['selected_idx'] = r['selected_idx']
        results['PMITS'][ld] = mean_r
        print(f"  LD={ld}: MAP={mean_r['MAP']*100:.2f}% (均值)")

    # 打印汇总表
    print("\n\n" + "=" * 75)
    print("MAP@900 汇总表 (%)")
    print("=" * 75)
    header = f"{'方法':<10}"
    for ld in LD_LIST:
        header += f" {'LD='+str(ld):>10}"
    print(header)
    print('─' * len(header))
    for name in ['PMITS'] + list(HASH_METHODS_2.keys()):
        row = f"{name:<10}"
        for ld in LD_LIST:
            row += f" {results[name][ld]['MAP']*100:>10.2f}"
        print(row)

    #  绘图
    print("\n生成对比图")
    plot_comparison(results, LD_LIST)
    plot_bar_comparison(results, LD_LIST)

    # 保存
    save_path = os.path.join(POOL_DIR, 'comparison_methods_2.npz')
    np.savez(save_path, results=results, ld_list=LD_LIST)
    print(f"结果已保存到 {save_path}")
    print("完成。")


if __name__ == '__main__':
    main()
