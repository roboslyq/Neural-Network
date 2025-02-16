from os.path import join, exists
from os import mkdir
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
"""
RNN自动编码器实现
目标：使用RNN实现序列的自动编码器，学习将序列压缩后还原

主要特点：
1. 使用编码器RNN将输入序列编码为隐藏状态
2. 使用解码器RNN从隐藏状态重建输入序列
3. 通过最小化重建误差来训练模型
"""

class RepeaterDataset(torch.utils.data.Dataset):
    """序列复制数据集
    生成随机的二进制序列，目标是重建相同的序列
    """
    def __init__(self, total_count, length):
        """初始化数据集
        Args:
            total_count: 样本总数
            length: 序列长度
        """
        # 生成随机的0-1序列
        self.data = torch.randint(0, 2, (total_count, length, 1)).type(torch.FloatTensor)
        # 目标就是输入序列本身
        self.labels = self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Datasets:
    """数据集管理类：创建训练集和测试集的数据加载器"""
    def __init__(self, batch_size, total_count=1000, length=15):
        # 创建训练数据加载器
        self.train_loader = torch.utils.data.DataLoader(
            RepeaterDataset(total_count, length), 
            batch_size=batch_size, shuffle=True)
        # 创建测试数据加载器
        self.test_loader = torch.utils.data.DataLoader(
            RepeaterDataset(total_count, length), 
            batch_size=batch_size * 2, shuffle=True)


class RNNAutoencoder(nn.Module):
    """RNN自动编码器模型
    使用两个RNN：一个编码器和一个解码器
    """
    def __init__(self, hidden_size=256):
        """初始化模型
        Args:
            hidden_size: RNN隐藏层的维度，默认256
        """
        super(RNNAutoencoder, self).__init__()
        self.hidden_size = hidden_size

        # 编码器RNN：将输入序列编码为隐藏状态
        self.rnn1 = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        # 解码器RNN：从隐藏状态重建序列
        self.rnn2 = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        # 输出层：将RNN输出映射到标量值
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x):
        """前向传播
        Args:
            x: 输入序列，形状为 [批次大小,序列长度,1]
        Returns:
            重建的序列
        """
        o1 = x.size()  # 获取输入形状
        o2 = self.rnn1(x)  # 编码器RNN
        # 创建全1序列作为解码器输入
        o3 = torch.ones(size=o1)  
        # 使用编码器的隐藏状态初始化解码器
        o4 = self.rnn2(o3, o2[1])  
        o5 = self.fc(o4[0])  # 输出层
        o6 = torch.sigmoid(o5)  # sigmoid激活
        return o6


class Trainer:
    """训练器类：处理模型训练、测试、保存和可视化"""
    def __init__(self, datasets, model, optimizer, loss_fn, results_path='results'):
        """初始化训练器
        Args:
            datasets: 数据集对象
            model: 神经网络模型
            optimizer: 优化器
            loss_fn: 损失函数
            results_path: 结果保存路径
        """
        self.datasets = datasets
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.results_path = results_path
        self.train_df = None

    def train_epoch(self, msg_format):
        self.model.train()

        losses = []
        bar = tqdm(self.datasets.train_loader)
        for data, target in bar:
            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.loss_fn(output, target)

            loss.backward()
            self.optimizer.step()

            bar.set_description(msg_format.format(loss.item()))

            losses.append(loss.item())
        return losses

    def test(self):
        self.model.eval()

        count = len(self.datasets.test_loader.dataset)
        length = self.datasets.test_loader.dataset[0][1].size()[0]
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.datasets.test_loader:
                output = self.model(data)
                test_loss += self.loss_fn(output, target).item() * len(data)
                correct += output.round().eq(target.round()).sum().item()

        return test_loss / count, correct / count / length

    def train(self, num_epoch):
        val_loss, accuracy = self.test()
        all_losses = [[None, val_loss, accuracy]]

        for epoch in range(num_epoch):
            # train
            train_losses = self.train_epoch(
                f'train {epoch}/{num_epoch} -- loss: {{:3.2f}}, val_loss: {val_loss:3.2f}, accuracy: {accuracy:.1%}')

            # test
            val_loss, accuracy = self.test()
            all_losses.extend([
                [train_loss, None, None]
                for train_loss in train_losses
            ])
            all_losses.append([None, val_loss, accuracy])

        self.save_model()
        self.train_df = pd.DataFrame(data=all_losses, columns=["train_loss", "val_loss", "accuracy"])
        self.train_df.to_csv(join(self.results_path, "train.csv"), index=False)

    def save_model(self):
        if not exists(self.results_path):
            mkdir(self.results_path)
        torch.save(self.model.state_dict(), join(self.results_path, 'model.pth'))

    def plot(self):
        import matplotlib.pyplot as plt
        self.train_df[["train_loss", "val_loss"]].ffill().plot(title="loss", grid=True, logy=False)
        self.train_df[["accuracy"]].dropna().plot(title="accuracy", grid=True)
        plt.show()


def train():
    """主训练函数"""
    torch.manual_seed(0)  # 设置随机种子
    length = 10  # 序列长度
    datasets = Datasets(100, total_count=5000, length=length)

    # 创建模型和训练组件
    model = RNNAutoencoder(hidden_size=5)
    loss_fn = torch.nn.MSELoss()  # 使用均方误差损失
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    trainer = Trainer(datasets, model=model, optimizer=optimizer,
                      loss_fn=loss_fn, results_path="results")

    # 训练模型
    trainer.train(num_epoch=1000)

    # 测试模型性能
    for i in range(5):
        # 生成随机测试序列
        test_x = torch.randint(0, 2, (1, length, 1)).type(torch.FloatTensor)
        test_y = test_x
        print(f"----- {i} -----")
        # 打印输入序列
        print(test_x.flatten().type(torch.IntTensor).tolist())
        # 预测并打印重建序列
        predict = model(test_x)
        print(predict.round().flatten().type(torch.IntTensor).tolist())
        # 打印重建准确率
        print(f"{predict.round().eq(test_y.round()).sum().item() / length:.0%}")

    trainer.plot()  # 绘制训练结果


if __name__ == "__main__":
    train()
