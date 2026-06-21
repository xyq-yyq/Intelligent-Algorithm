"""哈希池构建,整合了四种哈希方法，生成统一的 256 位哈希池。
四种哈希方法各提供 64 比特：
    比特   0– 63 : DeepHN  （深度哈希网络，有监督训练）
    比特  64–127 : HashNet （连续化哈希网络，有监督训练）
    比特 128–191 : IDeepHN （改进深度哈希，软相似度 + 有监督训练）
    比特 192–255 : LSH     （局部敏感哈希，随机投影，无监督）
"""

import os
import sys
import glob
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from hashing.DeepHN import train_deephn
from hashing.HashNet import train_hashnet
from hashing.IDeepHN import train_ideephn
from hashing.LSH import train_lsh, encode as encode_lsh

os.makedirs(POOL_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ResNet-50 图像预处理：缩放到 224×224，张量化，ImageNet 标准化
_RESNET_TRANSFORM = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 数据加载,从本地文件夹读取 CIFAR-10

def load_dataset():
    # 从 dataset/database/ 和 dataset/query/ 读取 CIFAR-10 图像。
    class_to_idx = {name: i for i, name in enumerate(CIFAR10_CLASSES)}

    def _load_split(split_name):
        # 加载一个数据分块（database 或 query）。
        root = os.path.join(DATA_DIR, split_name)
        imgs, labs = [], []
        for class_name in CIFAR10_CLASSES:
            class_dir = os.path.join(root, class_name)
            files = sorted(glob.glob(os.path.join(class_dir, '*.png')))
            for f in files:
                imgs.append(f)                        # 只存路径，用到时才解码
                labs.append(class_to_idx[class_name])
        return imgs, np.array(labs)

    train_paths, train_labs = _load_split('database')
    test_paths, test_labs = _load_split('query')

    # 按配置中的数量做子采样
    n_tr = min(NUM_TRAIN, len(train_paths))
    n_te = min(NUM_TEST, len(test_paths))
    idx_tr = np.random.RandomState(42).choice(len(train_paths), n_tr, replace=False)
    idx_te = np.random.RandomState(42).choice(len(test_paths), n_te, replace=False)

    return (list(np.array(train_paths)[idx_tr]), train_labs[idx_tr]), \
           (list(np.array(test_paths)[idx_te]), test_labs[idx_te])

# ResNet-50 特征提取

def extract_features(image_paths, labels, name='train'):
    # 用预训练 ResNet-50 提取图像特征。
    model = resnet50(weights=ResNet50_Weights.DEFAULT).to(DEVICE)
    model.fc = torch.nn.Identity()   # 移除分类头，保留特征
    model.eval()

    feats, labs = [], []
    n = len(image_paths)
    for i in range(0, n, BATCH_SIZE):
        batch = image_paths[i:i + BATCH_SIZE]
        tensors = []
        for item in batch:
            img = Image.open(item).convert('RGB') if isinstance(item, str) else item
            tensors.append(_RESNET_TRANSFORM(img))
        inp = torch.stack(tensors).to(DEVICE)
        with torch.no_grad():
            f = model(inp).cpu().numpy()
        feats.append(f)
        labs.append(labels[i:i + BATCH_SIZE])
        if (i // BATCH_SIZE + 1) % 50 == 0:
            print(f"  {name} 特征提取: {min(i + BATCH_SIZE, n)}/{n}")
    return np.vstack(feats), np.concatenate(labs)


# 哈希池完整构建

def build_hash_pool():
    # 构建完整的 256 位哈希池。
    import sys
    sys.stdout.reconfigure(line_buffering=True)  # 确保进度实时可见
    print(f"=== 构建哈希池 ===  设备: {DEVICE}")
    print(f"数据集: {DATASET}, 池大小: {LP} 比特")

    # 加载数据
    (train_paths, train_labs), (test_paths, test_labs) = load_dataset()
    print(f"训练集: {len(train_labs)}, 测试集: {len(test_labs)}")

    # 提取ResNet-50特征
    print("\n提取ResNet-50特征")
    train_feat, train_labs = extract_features(train_paths, train_labs, 'train')
    test_feat, test_labs = extract_features(test_paths, test_labs, 'test')
    print(f"训练特征: {train_feat.shape}, 测试特征: {test_feat.shape}")

    # 训练/生成四种哈希方法（各BITS_PER_METHOD)
    print(f"\n训练哈希方法（每种 {BITS_PER_METHOD} 比特，共 4 种）")

    # [1/4] DeepHN
    print("\n[1/4] 训练 DeepHN...")
    model_dh, codes_dh_train = train_deephn(
        train_feat, train_labs, BITS_PER_METHOD, epochs=30, device=DEVICE)
    from hashing.DeepHN import encode as encode_dh
    codes_dh_test = encode_dh(model_dh, test_feat, device=DEVICE)
    torch.save(model_dh.state_dict(), os.path.join(POOL_DIR, 'DeepHN_model.pth'))

    # [2/4] HashNet
    print("\n[2/4] 训练 HashNet...")
    model_hn, codes_hn_train = train_hashnet(
        train_feat, train_labs, BITS_PER_METHOD, epochs=30, device=DEVICE)
    from hashing.HashNet import encode as encode_hn
    codes_hn_test = encode_hn(model_hn, test_feat, device=DEVICE)
    torch.save(model_hn.state_dict(), os.path.join(POOL_DIR, 'HashNet_model.pth'))

    # [3/4] IDeepHN
    print("\n[3/4] 训练 IDeepHN...")
    model_id, codes_id_train = train_ideephn(
        train_feat, train_labs, BITS_PER_METHOD, epochs=30, device=DEVICE)
    from hashing.IDeepHN import encode as encode_id
    codes_id_test = encode_id(model_id, test_feat, device=DEVICE)
    torch.save(model_id.state_dict(), os.path.join(POOL_DIR, 'IDeepHN_model.pth'))

    # [4/4] LSH
    print("\n[4/4] LSH（随机投影）...")
    model_lsh, codes_lsh_train = train_lsh(train_feat, train_labs,
                                            BITS_PER_METHOD)
    codes_lsh_test = encode_lsh(model_lsh, test_feat)
    torch.save(model_lsh, os.path.join(POOL_DIR, 'LSH_model.pth'))

    # 拼接为统一哈希池
    # 池形状: (256, n) = (lp, n)
    pool_train = np.vstack([
        codes_dh_train.T,
        codes_hn_train.T,
        codes_id_train.T,
        codes_lsh_train.T,
    ]).astype(np.int8)
    pool_test = np.vstack([
        codes_dh_test.T,
        codes_hn_test.T,
        codes_id_test.T,
        codes_lsh_test.T,
    ]).astype(np.int8)

    print(f"\n哈希池形状，训练集: {pool_train.shape}, 测试集: {pool_test.shape}")
    print(f"训练特征: {train_feat.shape}")

    # 保存
    save_path = os.path.join(POOL_DIR, 'hash_pool.npz')
    np.savez_compressed(save_path,
                        pool_train=pool_train, pool_test=pool_test,
                        train_feat=train_feat.astype(np.float32),
                        test_feat=test_feat.astype(np.float32),
                        train_labels=train_labs, test_labels=test_labs,
                        has_lsh=True)
    print(f"\n哈希池已保存到: {save_path}")
    print("完成")

# 加载已有池

def load_hash_pool():
    # 加载预构建的哈希池文件。

    path = os.path.join(POOL_DIR, 'hash_pool.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"哈希池文件不存在: {path}\n"
            f"请先运行 'python hashing/hash_pool.py' 构建池。")
    data = np.load(path)
    return (data['pool_train'].astype(np.int8), data['pool_test'].astype(np.int8),
            data['train_feat'], data['test_feat'],
            data['train_labels'], data['test_labels'])


def load_hashing_models():
    # 加载已训练的哈希模型参数

    models = {}
    for name in ['DeepHN', 'HashNet', 'IDeepHN']:
        path = os.path.join(POOL_DIR, f'{name}_model.pth')
        if os.path.exists(path):
            models[name] = torch.load(path, map_location='cpu', weights_only=True)
    return models

# 入口

if __name__ == '__main__':
    build_hash_pool()
