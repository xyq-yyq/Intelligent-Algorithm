algorithm:
	ga_al.py:标准遗传算法（GA）。锦标赛选择 + 单点交叉 + 交换变异，用于哈希位选择。
	iga_al.py:改进型遗传算法（IGA）。集成禁忌搜索（TS）+ 路径重连（PR）+ 小生境技术，是 GA 的增强版。
	pmits_al.py:并行文化基因迭代禁忌搜索（PMITS），融合文化基因算法与禁忌搜索
	pso_adv_al.py:二进制粒子群优化（BPSO），带惯性权重衰减、重启策略、多样性保持。
	random_al.py:随机选择基线。随机抽取 ld 个比特，不做任何优化。
hashing：
	DeepHN.py:深度哈希网络（DeepHN）的训练与编码。
	hash_pool.py:构建 288 位哈希池：DeepHN + HashNet + IDeepHN + LSH（各 72 位）。
	hash_pool_2.py:构建 216 位传统哈希池：KLSH + PCAH + LSH（各 72 位）。
	HashNet.py:HashNet 的训练与编码。
	IDeepHN.py:改进深度哈希网络（IDeepHN）的训练与编码。
	ITQ.py:迭代量化哈希（ITQ）的训练与编码。
	KLSH.py:核化局部敏感哈希（KLSH）的训练与编码。
	LSH.py:局部敏感哈希（LSH）的训练与编码。
	PCAH.py:PCA 哈希（PCAH）的训练与编码。
test:
	compare_methods.py:PMITS 选择 vs 四种单一哈希方法（DeepHN、HashNet、IDeepHN、LSH）的对比实验。
	compare_methods_2.py:PMITS 选择 vs 三种传统哈希方法（KLSH、PCAH、LSH）的对比实验。
compare.py:主对比脚本。五种智能算法（Random、GA、PMITS、IGA、BPSO）在多个 LD 下的性能对比，生成指标图、耗时箱线图、收敛曲线图。
config.py:全局配置文件。数据集路径、哈希池参数、LD_LIST、算法参数、随机种子等。
utils.py:工具函数库。亲和度矩阵计算、相似度矩阵、拉普拉斯矩阵、互信息、检索指标（MAP、Precision、Recall、PH2）等。
