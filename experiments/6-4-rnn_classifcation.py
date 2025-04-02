from os.path import join, exists
from os import mkdir
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
"""
RNN分类实现
"""

def obj_func(xs):
    """计算序列中0和1的数量并进行分类
    Args:
        xs: 包含0和1的列表
    Returns:
        int: 返回分类结果
            2: 1的数量大于0的数量
            1: 1的数量等于0的数量
            0: 1的数量小于0的数量
    """
    num_ones = sum(xs)  # 计算1的数量
    num_zeros = len(xs) - num_ones  # 计算0的数量
    return 2 if num_ones > num_zeros else (1 if num_ones == num_zeros else 0)


class ClassificationDataset(torch.utils.data.Dataset):
    """分类任务数据集
    生成随机的0-1序列，并根据序列中0和1的数量关系进行分类
    """
    def __init__(self, total_count, length):
        """初始化数据集
        Args:
            total_count: 样本总数
            length: 每个序列的长度
        """
        # 生成随机的0-1序列，形状为(样本数, 序列长度, 1)
        self.data = torch.randint(0, 2, (total_count, length, 1)).type(torch.FloatTensor)
        # 计算每个序列的标签（0,1,2三类）
        self.labels = torch.LongTensor([
            obj_func(xs.flatten().tolist())
            for xs in self.data
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Datasets:
    """数据集管理类
    创建训练集和测试集的数据加载器
    """
    def __init__(self, batch_size, total_count=1000, length=15):
        """初始化数据加载器
        Args:
            batch_size: 批次大小
            total_count: 数据集样本总数，默认1000
            length: 序列长度，默认15
        """
        # 创建训练数据加载器
        self.train_loader = torch.utils.data.DataLoader(
            ClassificationDataset(total_count, length), 
            batch_size=batch_size,
            shuffle=True
        )
        # 创建测试数据加载器，batch_size是训练集的两倍
        self.test_loader = torch.utils.data.DataLoader(
            ClassificationDataset(total_count, length), 
            batch_size=batch_size * 2,
            shuffle=True
        )


class RNNClassifier(nn.Module):
    """RNN分类器模型
    使用LSTM和全连接层实现序列分类
    """
    def __init__(self, hidden_size=10):
        """初始化模型
        Args:
            hidden_size: LSTM隐藏层的维度，默认10
        """
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        # LSTM层，输入维度为1，隐藏层维度为hidden_size
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        # 全连接层，将LSTM的输出映射到3个类别
        self.fc = nn.Linear(in_features=hidden_size, out_features=3)

    def forward(self, x):
        """前向传播函数
        Args:
            x: 输入张量，形状为 [批次大小,序列长度,1]
        Returns:
            对数概率分布，形状为 [批次大小,类别数]
        """
        o1 = self.lstm(x)  # LSTM处理序列
        o2 = torch.squeeze(o1[1][0])  # 获取最后一个时间步的隐藏状态
        o3 = self.fc(o2)  # 全连接层分类
        o4 = torch.log_softmax(o3, dim=-1)  # 计算对数概率
        return o4


class Trainer:
    """训练器类
    处理模型训练、测试、保存和结果可视化
    """
    def __init__(self, datasets, model, optimizer, loss_fn, results_path='results'):
        """初始化训练器
        Args:
            datasets: 包含训练和测试数据加载器的数据集对象
            model: 要训练的神经网络模型
            optimizer: 优化器对象
            loss_fn: 损失函数
            results_path: 结果保存路径，默认为'results'
        """
        self.datasets = datasets
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.results_path = results_path
        self.train_df = None

    def train_epoch(self, msg_format):
        """训练一个完整的epoch
        Args:
            msg_format: 进度条显示的消息格式字符串
        Returns:
            losses: 列表，包含这个epoch中每个batch的损失值
        """
        self.model.train()  # 设置为训练模式
        losses = []
        bar = tqdm(self.datasets.train_loader)
        for data, target in bar:
            self.optimizer.zero_grad()  # 清除梯度
            output = self.model(data)  # 前向传播
            loss = self.loss_fn(output, target)  # 计算损失
            loss.backward()  # 反向传播
            self.optimizer.step()  # 更新参数
            bar.set_description(msg_format.format(loss.item()))  # 更新进度条
            losses.append(loss.item())
        return losses

    def test(self):
        """在测试集上评估模型
        Returns:
            test_loss: float, 平均测试损失
            accuracy: float, 测试准确率
        """
        self.model.eval()  # 设置为评估模式
        count = len(self.datasets.test_loader.dataset)
        test_loss = 0
        correct = 0
        with torch.no_grad():  # 不计算梯度
            for data, target in self.datasets.test_loader:
                output = self.model(data)  # 前向传播
                # 累加批次损失
                test_loss += self.loss_fn(output, target).item() * len(data)
                # 计算正确预测的数量
                correct += output.argmax(dim=-1).eq(target).sum().item()

        return test_loss / count, correct / count

    def train(self, num_epoch):
        """训练模型指定的轮数
        Args:
            num_epoch: int, 要训练的总epoch数
        """
        val_loss, accuracy = self.test()  # 初始测试
        all_losses = [[None, val_loss, accuracy]]

        for epoch in range(num_epoch):
            # 训练一个epoch
            train_losses = self.train_epoch(
                f'train {epoch}/{num_epoch} -- loss: {{:3.2f}}, val_loss: {val_loss:3.2f}, accuracy: {accuracy:.1%}')

            # 测试当前模型性能
            val_loss, accuracy = self.test()
            # 记录所有损失值
            all_losses.extend([
                [train_loss, None, None]
                for train_loss in train_losses
            ])
            all_losses.append([None, val_loss, accuracy])

        self.save_model()  # 保存模型
        # 保存训练历史为CSV文件
        self.train_df = pd.DataFrame(data=all_losses, columns=["train_loss", "val_loss", "accuracy"])
        self.train_df.to_csv(join(self.results_path, "train.csv"), index=False)

    def save_model(self):
        """保存模型参数到文件"""
        if not exists(self.results_path):
            mkdir(self.results_path)  # 创建保存目录
        torch.save(self.model.state_dict(), join(self.results_path, 'model.pth'))

    def plot(self):
        """绘制训练过程的损失和准确率曲线"""
        import matplotlib.pyplot as plt
        # 绘制损失曲线
        self.train_df[["train_loss", "val_loss"]].ffill().plot(title="loss", grid=True, logy=False)
        # 绘制准确率曲线
        self.train_df[["accuracy"]].dropna().plot(title="accuracy", grid=True)
        plt.show()


def train():
    """主训练函数"""
    torch.manual_seed(0)  # 设置随机种子
    datasets = Datasets(100, total_count=5000, length=20)  # 初始化数据集

    model = RNNClassifier()  # 创建RNN分类器模型

    # 设置损失函数（负对数似然损失）和优化器
    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    
    # 创建训练器并开始训练
    trainer = Trainer(datasets, model=model, optimizer=optimizer,
                      loss_fn=loss_fn, results_path="results")

    trainer.train(num_epoch=30)  # 训练30个epoch
    trainer.plot()  # 绘制训练结果


if __name__ == "__main__":
    train()
