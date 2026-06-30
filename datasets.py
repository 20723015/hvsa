import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np


def load_data(dataset_name, data_dir='./data'):
    """下载并加载指定的数据集"""
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    elif dataset_name == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

    else:
        raise ValueError(f"未知的数据集: {dataset_name}")

    return train_dataset, test_dataset


# =========================================================================
# 重点检查下面这一行！必须包含 max_samples_per_user=None
# =========================================================================
def distribute_data(train_dataset, num_users, batch_size, iid=True,
                    max_samples_per_user=None, dirichlet_alpha=0.5):
    """
    将训练数据分配给多个用户。

    Args:
        train_dataset: 训练数据集。
        num_users: 用户数量。
        batch_size: 每个 DataLoader 的 batch_size。
        iid: True=IID均匀分配, False=Non-IID Dirichlet 分配。
        max_samples_per_user: 每个用户最大样本数 (可选)。
        dirichlet_alpha: Dirichlet 分布浓度参数 (iid=False 时生效)。
                        α→0: 极端Non-IID (每个用户只有1-2类)
                        α=0.5: 中度Non-IID
                        α=1.0: 轻度Non-IID
                        α→∞: 趋近IID
    """
    # 提取标签
    targets = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    num_classes = len(np.unique(targets))

    if iid:
        # --- IID: 均匀随机分配 ---
        all_indices = np.random.permutation(len(train_dataset))
        if max_samples_per_user is not None:
            total_needed = num_users * max_samples_per_user
            if total_needed <= len(train_dataset):
                all_indices = all_indices[:total_needed]
        num_items = len(all_indices) // num_users
        user_loaders = []
        for i in range(num_users):
            start, end = i * num_items, (i + 1) * num_items if i < num_users - 1 else len(all_indices)
            subset = Subset(train_dataset, all_indices[start:end])
            loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
            user_loaders.append(loader)
        print(f"[IID]  {num_users} 用户, 每用户约 {num_items} 条")
        return user_loaders
    else:
        # --- Non-IID: Dirichlet 分布 ---
        # 对每个类别，用 Dirichlet 分配到 n 个用户
        class_indices = [np.where(targets == c)[0] for c in range(num_classes)]
        user_indices = [[] for _ in range(num_users)]

        for c in range(num_classes):
            np.random.shuffle(class_indices[c])
            # Dirichlet 分配比例
            proportions = np.random.dirichlet([dirichlet_alpha] * num_users)
            # 按比例分配该类样本
            proportions = proportions / proportions.sum()
            cumsum = np.cumsum(proportions)
            cumsum[-1] = 1.0

            class_size = len(class_indices[c])
            if max_samples_per_user is not None:
                max_class_per_user = max_samples_per_user // num_classes
                class_size = min(class_size, max_class_per_user * num_users)

            starts = (cumsum * class_size).astype(int)
            starts = np.concatenate([[0], starts])
            for u in range(num_users):
                s, e = starts[u], starts[u + 1] if u < num_users - 1 else class_size
                if s < e and s < len(class_indices[c]):
                    user_indices[u].extend(class_indices[c][s:e])

        # 检查是否有用户为空
        for u in range(num_users):
            if len(user_indices[u]) == 0:
                user_indices[u] = [np.random.randint(len(train_dataset))]

        # 构建 DataLoader
        user_loaders = []
        for u in range(num_users):
            np.random.shuffle(user_indices[u])
            subset = Subset(train_dataset, user_indices[u])
            loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
            user_loaders.append(loader)

        # 统计分布
        class_counts = []
        for u in range(num_users):
            counts = [np.sum(targets[user_indices[u]] == c) for c in range(num_classes)]
            class_counts.append(counts)
        avg_per_user = np.mean([len(idx) for idx in user_indices])

        print(f"[Non-IID α={dirichlet_alpha}] {num_users} 用户, "
              f"每用户约 {avg_per_user:.0f} 条, "
              f"最少 {min(len(idx) for idx in user_indices)} / 最多 {max(len(idx) for idx in user_indices)}")
        return user_loaders