import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader


# --- 1. 定义模型 ---

class Tiny_MNIST(nn.Module):
    """极小线性模型 (dim=7850).

    仅用于安全验证线的全密码学端到端实验: LHH.Hash 为逐维 EC 标量乘
    (实测 ~165us/维), 7850 维单次 Hash ~1.3s, 端到端 verify 实验可承受;
    全维 CNN (582026) 单次 Hash ~96s, 不可行。检测/验证是代数性质,
    与维度无关 (任一维篡改即破坏等式), 故功能结论可安全外推。
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = self.flatten(x)
        return F.log_softmax(self.fc(x), dim=1)


class MLP_MNIST(nn.Module):
    """适用于 MNIST / FashionMNIST 的 MLP"""

    def __init__(self):
        super(MLP_MNIST, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class CNN_MNIST(nn.Module):
    """适用于 MNIST / FashionMNIST 的 CNN"""

    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)  # 4*4 是因为 (28-4)/2 = 12, (12-4)/2 = 4
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNN_CIFAR10(nn.Module):
    """适用于 CIFAR-10 的 CNN"""

    def __init__(self):
        super(CNN_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 5 * 5, 512)  # 5*5 是因为 (32-4)/2 = 14, (14-4)/2 = 5
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def get_model(dataset_name, model_name):
    """根据名称获取模型实例"""
    if dataset_name in ['MNIST', 'FashionMNIST']:
        if model_name == 'MLP':
            return MLP_MNIST()
        elif model_name == 'Tiny':
            return Tiny_MNIST()
        elif model_name == 'CNN':
            return CNN_MNIST()
    elif dataset_name == 'CIFAR10':
        if model_name == 'CNN':
            return CNN_CIFAR10()
        elif model_name == 'MLP':
            raise ValueError("CIFAR10 不支持 MLP (请使用 CNN)")
    raise ValueError(f"未知的模型/数据集组合: {dataset_name}, {model_name}")


# --- 2. 训练和评估辅助函数 ---

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_local(model, train_loader, epochs, lr, prox_mu=0.0,
                curriculum='none', dpr_q=0.0):
    """标准的本地训练循环.

    FedProx: prox_mu>0 时对每个 batch 的损失加近端项 (mu/2)·||w - w0||²,
    锚点 w0 是调用方 unflatten 进来的全局基模型 (即组包基版本 r_k 上的 W)。
    纯本地损失项: 不新增消息、不改变上传的加性掩码结构。

    FedBSS 课程化 (AAAI 2025, arXiv:2501.11360 的单 epoch 适配): curriculum=
    bias_asc/bias_desc 时, 先在训练起点模型下逐样本评估 NLL 偏差, 按 偏差升序
    (bias_asc, 小偏差先训) / 降序 (bias_desc, 方向对照) 组织 batch。
    RNG 流保持: 基线每 epoch 迭代一次 shuffle loader (消耗全局 torch RNG 的
    播种抽取); 此处同样每 epoch 做一次丢弃迭代补偿, 偏差评估用手动小批前向
    (不创建 DataLoader 迭代器) —— 全程抽取次数与基线逐值一致, 跨批次配对精确。

    DPR 参数重置 (NeurIPS 2025 Dynamic Parameter Reset 的适配): dpr_q>0 时,
    每步更新后把 |w−w0| 超过其 q 分位数的维度重置回全局锚 w0 (硬版 prox)。
    """
    model.to(DEVICE)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.NLLLoss()
    w0 = None
    if prox_mu > 0 or dpr_q > 0:
        with torch.no_grad():
            w0 = [p.detach().clone() for p in model.parameters()]

    def _dpr_reset():
        if dpr_q <= 0:
            return
        with torch.no_grad():
            for p, p0 in zip(model.parameters(), w0):
                drift = (p - p0).abs()
                thresh = torch.quantile(drift.flatten(), dpr_q)
                mask = drift > thresh
                p[mask] = p0[mask]

    def _step(x, y):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        if prox_mu > 0:
            prox = sum((p - p0).pow(2).sum()
                       for p, p0 in zip(model.parameters(), w0))
            loss = loss + (prox_mu / 2.0) * prox
        loss.backward()
        optimizer.step()
        _dpr_reset()

    ds = train_loader.dataset
    bs = train_loader.batch_size or 64
    if curriculum in ('bias_asc', 'bias_desc'):
        # 每 epoch 一次丢弃迭代: 与基线的 loader 播种抽取次数逐值一致
        for _ in range(epochs):
            for _ in train_loader:
                break
        n = len(ds)
        model.eval()
        losses = torch.empty(n)
        with torch.no_grad():
            chunk = 256
            for s in range(0, n, chunk):
                idx = list(range(s, min(s + chunk, n)))
                x = torch.stack([ds[i][0] for i in idx]).to(DEVICE)
                y = torch.tensor([ds[i][1] for i in idx]).to(DEVICE)
                out = model(x)
                losses[s:s + len(idx)] = F.nll_loss(out, y, reduction='none').cpu()
        order = torch.argsort(losses)          # 小偏差在前
        if curriculum == 'bias_desc':
            order = order.flip(0)              # 方向对照: 大偏差在前
        model.train()
        for start in range(0, n, bs):
            idx = order[start:start + bs].tolist()
            x = torch.stack([ds[i][0] for i in idx]).to(DEVICE)
            y = torch.tensor([ds[i][1] for i in idx]).to(DEVICE)
            _step(x, y)
    else:
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                _step(data, target)
    model.to('cpu')  # 将模型移回 CPU，以便参数提取


def evaluate(model, test_loader):
    """标准的评估循环"""
    model.to(DEVICE)
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.NLLLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    model.to('cpu')
    return accuracy, test_loss


# --- 3. 展平和解平参数 (安全聚合的关键) ---

def flatten_parameters(model):
    """将模型所有参数展平到一个 Numpy 向量中"""
    vectors = []
    model_meta = []  # 存储形状和大小，用于解平

    for param in model.parameters():
        param_data = param.data.cpu().numpy()
        vectors.append(param_data.flatten())
        model_meta.append((param_data.shape, param_data.size))

    flat_vector = np.concatenate(vectors)
    return flat_vector, model_meta


def unflatten_parameters(model, flat_vector, model_meta):
    """
    使用展平的 Numpy 向量和元数据来恢复模型参数
    """
    model.to('cpu')
    idx = 0
    for param, (shape, size) in zip(model.parameters(), model_meta):
        # 提取切片
        chunk = flat_vector[idx: idx + size]
        # 恢复形状并转换为 tensor
        param_data = torch.from_numpy(chunk.reshape(shape).copy())
        # 加载回模型
        param.data.copy_(param_data)
        idx += size