"""全局配置文件，所有模块的统一参数入口。
本文件定义数据集路径、哈希池规模、亲和度矩阵超参数、各算法的默认参数、评估指标设定等。
"""

import os

# 路径配置

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # 项目根目录
DATA_DIR = os.path.join(BASE_DIR, 'dataset')             # CIFAR-10 数据集目录
POOL_DIR = os.path.join(BASE_DIR, 'pool')                # 哈希池及结果输出目录

# 数据集配置（CIFAR-10）

DATASET = 'cifar10'
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
NUM_TRAIN = 50000      # 训练样本数（每类 5000）
NUM_TEST = 10000       # 测试样本数（每类 1000）
NUM_QUERY = 1000       # 检索评估用的查询样本数

# ResNet-50 特征提取=

FEATURE_DIM = 2048     # 全连接层之前的特征维度
BATCH_SIZE = 64        # 特征提取时的批大小

#0– 71 : DeepHN
#72–143 : HashNet
#144–215 : IDeepHN
#216–287 : LSH

BITS_PER_METHOD = 72              # 每种哈希方法生成的比特数
LP = BITS_PER_METHOD * 4          # 哈希池总大小 = 288

# 亲和度矩阵

XI = 2.0
BETA = 2.0
K_NEIGHBORS = 100                 # kNN 相似度图的近邻数

# 遗传算法（GA）参数

GA_POP_SIZE = 100                 # 种群大小
GA_GENERATIONS = 200              # 进化代数
GA_CROSSOVER_PROB = 0.8           # 交叉概率
GA_MUTATION_PROB = 0.05           # 变异概率

# 随机选择（Random）基线参数

RANDOM_TRIALS = 5000              # 随机尝试次数

# 随机种子用于结果复现

SEED = 42

# 评估参数

LD_LIST = [8, 16, 24, 32, 40, 48, 56, 64, 72]   # 待测试的目标哈希码长度列表
TOP_NK = 900                                  # Precision@NK / Recall@NK 的 K 值
HAMMING_R = 2                                 # 汉明半径（用于 PH2 等指标）
