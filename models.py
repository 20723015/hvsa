import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader


# --- 1. 定义模型 ---

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


def train_local(model, train_loader, epochs, lr):
    """标准的本地训练循环"""
    model.to(DEVICE)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
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