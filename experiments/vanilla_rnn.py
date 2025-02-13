from os.path import join, exists
from os import mkdir
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn


def obj_func(xs):
    """计算累积和函数
    Args:
        xs: [L] 一个整数列表，包含0和1
    Returns:
        [L] 返回一个列表，每个位置存储从开始到当前位置的累积和
    """
    results = []
    _c = 0
    for x in xs:
        _c += x.item()  # 累加当前值
        results.append(_c)  # 将累积和添加到结果列表
    return results


class CounterDataset(torch.utils.data.Dataset):
    """计数数据集类
    生成随机的0-1序列及其对应的累积和
    """
    def __init__(self, total_count, length):
        """初始化数据集
        Args:
            total_count: 样本总数
            length: 每个序列的长度
        """
        # 生成随机的0-1序列，形状为(样本数, 序列长度, 1)
        self.data = torch.randint(0, 2, (total_count, length, 1)).type(torch.FloatTensor)
        # 计算每个序列的累积和，并增加一个维度
        self.labels = torch.unsqueeze(torch.FloatTensor([
            obj_func(xs.flatten())
            for xs in self.data
        ]), -1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Datasets:
    """数据集管理类
    创建训练集和测试集的数据加载器
    """
    def __init__(self, batch_size, total_count=1000, length=15):
        # 创建训练数据加载器
        self.train_loader = torch.utils.data.DataLoader(
            CounterDataset(total_count, length), 
            batch_size=batch_size,
            shuffle=True
        )
        # 创建测试数据加载器，batch_size是训练集的两倍
        self.test_loader = torch.utils.data.DataLoader(
            CounterDataset(total_count, length), 
            batch_size=batch_size * 2,
            shuffle=True
        )


class VanillaRNN(nn.Module):
    """基础RNN模型类
    使用PyTorch的RNN实现一个简单的循环神经网络
    """
    def __init__(self, hidden_size=1):
        """初始化RNN模型
        Args:
            hidden_size: 隐藏状态的维度，默认为1
        """
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        # 创建RNN层，输入维度为1，使用ReLU激活函数
        self.rnn = nn.RNN(
            input_size=1, 
            hidden_size=hidden_size, 
            nonlinearity='relu', 
            batch_first=True
        )

    def forward(self, x):
        """前向传播函数
        Args:
            x: 输入张量，形状为 [批次大小,序列长度,1]
        Returns:
            RNN的输出序列
        """
        o1 = self.rnn(x)
        return o1[0]  # 返回输出序列，不返回隐藏状态


class Trainer:
    """训练器类
    用于管理模型的训练、测试、保存和可视化过程
    """
    def __init__(self, datasets, model, optimizer, loss_fn, results_path='results'):
        """初始化训练器
        Args:
            datasets: 包含训练和测试数据加载器的数据集对象
            model: 要训练的神经网络模型
            optimizer: 优化器对象，用于更新模型参数
            loss_fn: 损失函数
            results_path: 模型和训练结果的保存路径，默认为'results'
        """
        self.datasets = datasets
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.results_path = results_path
        self.train_df = None  # 用于存储训练过程中的损失和准确率

    def train_epoch(self, msg_format):
        """训练一个完整的epoch
        Args:
            msg_format: 进度条显示的消息格式字符串
        Returns:
            losses: 列表，包含这个epoch中每个batch的损失值
        """
        self.model.train()  # 将模型设置为训练模式

        losses = []
        bar = tqdm(self.datasets.train_loader)  # 创建进度条
        for data, target in bar:
            self.optimizer.zero_grad()  # 清除之前的梯度

            output = self.model(data)  # 前向传播
            loss = self.loss_fn(output, target)  # 计算损失

            loss.backward()  # 反向传播计算梯度
            self.optimizer.step()  # 更新模型参数

            # 更新进度条显示的信息
            bar.set_description(msg_format.format(loss.item()))
            losses.append(loss.item())  # 记录当前batch的损失值
        return losses

    def test(self):
        """在测试集上评估模型性能
        Returns:
            test_loss: float, 平均测试损失
            correct / count / length: float, 测试准确率（正确预测的比例）
        """
        self.model.eval()  # 将模型设置为评估模式

        count = len(self.datasets.test_loader.dataset)  # 测试集样本总数
        length = self.datasets.test_loader.dataset[0][1].size()[0]  # 序列长度
        test_loss = 0
        correct = 0
        
        with torch.no_grad():  # 测试时不需要计算梯度
            for data, target in self.datasets.test_loader:
                output = self.model(data)  # 前向传播
                # 累加批次损失
                test_loss += self.loss_fn(output, target).item() * len(data)
                # 计算正确预测的数量（将预测值四舍五入后与目标值比较）
                correct += output.round().eq(target.round()).sum().item()

        return test_loss / count, correct / count / length

    def train(self, num_epoch):
        """训练模型指定的轮数
        Args:
            num_epoch: int, 要训练的总epoch数
        """
        # 在训练开始前进行初始测试
        val_loss, accuracy = self.test()
        all_losses = [[None, val_loss, accuracy]]

        for epoch in range(num_epoch):
            # 训练一个epoch并获取训练损失
            train_losses = self.train_epoch(
                f'train {epoch}/{num_epoch} -- loss: {{:3.2f}}, val_loss: {val_loss:3.2f}, accuracy: {accuracy:.1%}')

            # 在当前epoch结束后进行测试
            val_loss, accuracy = self.test()
            
            # 记录所有训练损失和测试结果
            all_losses.extend([
                [train_loss, None, None]
                for train_loss in train_losses
            ])
            all_losses.append([None, val_loss, accuracy])

        # 保存训练好的模型
        self.save_model()
        
        # 将训练历史记录保存为CSV文件
        self.train_df = pd.DataFrame(data=all_losses, columns=["train_loss", "val_loss", "accuracy"])
        self.train_df.to_csv(join(self.results_path, "train.csv"), index=False)

    def save_model(self):
        """保存模型参数到文件
        如果保存目录不存在，会先创建目录
        """
        if not exists(self.results_path):
            mkdir(self.results_path)  # 创建保存目录
        # 保存模型状态字典
        torch.save(self.model.state_dict(), join(self.results_path, 'model.pth'))

    def plot(self):
        """绘制训练过程的损失和准确率曲线
        使用matplotlib创建两个子图：
        1. 训练损失和验证损失的变化曲线
        2. 模型准确率的变化曲线
        """
        import matplotlib.pyplot as plt
        # 绘制损失曲线（使用前向填充处理缺失值）
        self.train_df[["train_loss", "val_loss"]].ffill().plot(title="loss", grid=True, logy=False)
        # 绘制准确率曲线（删除缺失值）
        self.train_df[["accuracy"]].dropna().plot(title="accuracy", grid=True)
        plt.show()


def train():
    """主训练函数"""
    torch.manual_seed(3)  # 设置随机种子以确保结果可重现
    datasets = Datasets(100, total_count=1000)  # 初始化数据集

    model = VanillaRNN()  # 创建RNN模型

    # 设置损失函数和优化器
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 创建训练器并开始训练
    trainer = Trainer(datasets, model=model, optimizer=optimizer,
                      loss_fn=loss_fn, results_path="results")

    trainer.train(num_epoch=40)
    
    # 打印模型参数
    params = list(model.parameters())
    print(params)
    # 提取RNN的权重和偏置参数
    w1 = params[0][0][0].item()      # 输入权重
    w2 = params[1][0][0].item()      # 隐藏状态权重
    b1 = params[2][0].item()         # 输入偏置
    b2 = params[3][0].item()         # 隐藏状态偏置
    # 打印RNN的数学表达式
    print(f"relu({w1:.3f}x{b1:+.3f}{w2:+.3f}h{b2:+.3f})")

    trainer.plot()  # 绘制训练过程的损失和准确率曲线


if __name__ == "__main__":
    train()
