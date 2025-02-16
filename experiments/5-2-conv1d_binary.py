from os.path import join
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
"""
一维卷积实验1：二元特征序列检测
目标：使用一维卷积网络检测序列中的"10"模式
例如：输入序列"1110101"，输出"001010"（检测每个位置是否出现从1到0的下降）

任务目标：
    检测二进制序列中的"10"模式（从1到0的下降）
    输入：随机生成的0-1序列
    输出：标记序列中每个位置是否出现"10"模式
网络结构：
    使用一维卷积层（kernel_size=2）检测相邻位置的模式
    sigmoid激活函数将输出映射到[0,1]区间
数据处理：
    生成随机的二进制序列作为输入
    使用obj_func计算目标标签
    实现了自定义的Dataset类处理数据
4. 训练过程：
    使用MSE损失函数
    Adam优化器
    训练10个epoch
    包含完整的训练、测试和可视化流程
创新点：
    使用一维卷积实现序列模式检测
    简单而有效的网络结构
    完整的训练框架
这是一个很好的一维卷积应用示例，展示了如何使用CNN处理序列数据。
"""

class Net1d(nn.Module):
    """一维卷积神经网络模型
    使用kernel_size=2的一维卷积来检测相邻位置的模式
    """
    def __init__(self):
        super(Net1d, self).__init__()
        # 创建一维卷积层：
        # in_channels=1: 输入序列只有一个通道
        # out_channels=1: 输出一个通道
        # kernel_size=2: 卷积核大小为2，用于检测相邻两个位置
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2)

    def forward(self, x):
        """前向传播函数
        Args:
            x: 输入张量，形状为 [批次大小,1,序列长度]
        Returns:
            经过sigmoid激活的卷积输出
        """
        o1 = self.conv(x)  # 一维卷积运算
        o2 = torch.sigmoid(o1)  # sigmoid激活函数
        return o2


def obj_func(in_tensor):
    """目标函数：检测序列中的"10"模式
    检查每个位置i，如果位置i是1且i+1是0，则输出1，否则输出0
    
    Args:
        in_tensor: 输入张量，形状为 [批次大小,1,序列长度]
    Returns:
        输出张量，形状为 [批次大小,1,序列长度-1]，标记了每个位置是否出现"10"模式
    """
    return torch.FloatTensor([
        [
            [
                1 if seq[i + 1] < x else 0  # 如果后一个数小于当前数（即出现下降），输出1
                for i, x in enumerate(seq[:-1])
            ]
            for seq in xs
        ]
        for xs in in_tensor
    ])


class PairUpDataset(torch.utils.data.Dataset):
    """数据集类：生成随机的二进制序列及其标签"""
    def __init__(self, total_count, length):
        """初始化数据集
        Args:
            total_count: 样本总数
            length: 序列长度
        """
        # 生成随机的0-1序列
        self.data = torch.randint(0, 2, (total_count,1,length)).type(torch.FloatTensor)
        # 计算目标标签（检测"10"模式）
        self.labels = obj_func(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Datasets:
    """数据集管理类：创建训练集和测试集的数据加载器"""
    def __init__(self, batch_size, total_count=1000, length=10):
        """初始化数据加载器
        Args:
            batch_size: 批次大小
            total_count: 数据集样本总数，默认1000
            length: 序列长度，默认10
        """
        # 创建训练数据加载器
        self.train_loader = torch.utils.data.DataLoader(
          PairUpDataset(total_count, length),
          batch_size=batch_size, shuffle=True)
        # 创建测试数据加载器
        self.test_loader = torch.utils.data.DataLoader(
          PairUpDataset(total_count, length),
          batch_size=batch_size * 2, shuffle=True)


class Trainer:
    """训练器类：处理模型训练、测试、保存和可视化"""
    def __init__(self, datasets, model, optimizer, loss_fn, results_path='D:\\ai\\models'):
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
        """训练一个epoch
        Args:
            msg_format: 进度条显示的消息格式
        Returns:
            losses: 每个batch的损失值列表
        """
        self.model.train()  # 设置为训练模式
        losses = []
        bar = tqdm(self.datasets.train_loader)
        for data, target in bar:
            self.optimizer.zero_grad()  # 清除梯度
            output = self.model(data)   # 前向传播
            loss = self.loss_fn(output, target)  # 计算损失
            loss.backward()  # 反向传播
            self.optimizer.step()  # 更新参数
            bar.set_description(msg_format.format(loss.item()))
            losses.append(loss.item())
        return losses

    def test(self):
        """测试模型性能
        Returns:
            test_loss: 平均测试损失
            accuracy: 测试准确率
        """
        self.model.eval()  # 设置为评估模式
        count = len(self.datasets.test_loader.dataset)
        length = self.datasets.test_loader.dataset[0][1].size()[-1]
        test_loss = 0
        correct = 0
        with torch.no_grad():  # 不计算梯度
            for data, target in self.datasets.test_loader:
                output = self.model(data).gt(0.5)  # 预测值大于0.5判定为1
                test_loss += self.loss_fn(output, target).item() * len(data)
                correct += output.eq(target).sum().item()  # 计算正确预测的数量

        return test_loss / count, correct / count / length

    def train(self, num_epoch):
        """训练模型
        Args:
            num_epoch: 训练轮数
        """
        val_loss, accuracy = self.test()  # 初始测试
        all_losses = [[None, val_loss, accuracy]]
        # 使用完整的数据集训练一次，叫做1个epoch。通常会使用完整的数据集训练多次
        for epoch in range(num_epoch):
            # 训练一个epoch
            train_losses = self.train_epoch(
                f'train {epoch}/{num_epoch} -- loss: {{:3.2f}}, val_loss: {val_loss:3.2f}, accuracy: {accuracy:.1%}')

            # 测试模型性能
            val_loss, accuracy = self.test()
            # 记录所有损失值
            all_losses.extend([
                [train_loss, None, None]
                for train_loss in train_losses
            ])
            all_losses.append([None, val_loss, accuracy])

        self.save_model()  # 保存模型
        # 保存训练历史
        self.train_df = pd.DataFrame(data=all_losses, columns=["train_loss", "val_loss", "accuracy"])
        self.train_df.to_csv(join(self.results_path, "train.csv"), index=False)

    def save_model(self):
        """保存模型参数"""
        torch.save(self.model.state_dict(), join(self.results_path, 'model.pth'))

    def plot(self):
        """绘制训练过程的损失和准确率曲线"""
        import matplotlib.pyplot as plt
        self.train_df[["train_loss", "val_loss"]].ffill().plot(grid=True, logy=True)
        self.train_df[["accuracy"]].dropna().plot(grid=True)
        plt.show()


def train():
    """主训练函数"""
    torch.manual_seed(0)  # 设置随机种子

    # 创建模型和训练组件
    model = Net1d()
    loss_fn = torch.nn.MSELoss()  # 使用均方误差损失
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)  # Adam优化器
    trainer = Trainer(Datasets(100), model=model, optimizer=optimizer,
                      loss_fn=loss_fn, results_path="D:\\ai\\models")

    # 训练模型
    trainer.train(num_epoch=10)
    print(list(model.parameters()))  # 打印模型参数
    trainer.plot()  # 绘制训练结果


if __name__ == "__main__":
    train()
