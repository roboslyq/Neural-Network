from os.path import join
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn


class FC1Model(nn.Module):
    """单层全连接神经网络模型
    包含一个线性层，将15维输入转换为50维输出，然后通过sigmoid激活函数
    """
    def __init__(self):
        super(FC1Model, self).__init__()
        # 定义一个全连接层，输入特征15维，输出特征50维
        self.fc = nn.Linear(in_features=15, out_features=50)

    def forward(self, x):
        """前向传播函数
        Args:
            x: 输入张量，形状为 [批次大小,1,15]
        Returns:
            经过处理后的一维输出张量
        """
        o1 = self.fc(x)                  # 第1步：线性变换
        o2 = torch.sigmoid(o1)           # 第2步：sigmoid激活
        o3 = torch.max(o2, dim=-1)[0]    # 第3步：在最后一个维度上取最大值
        o4 = o3.reshape(-1)              # 展平成一维张量
        # 此处可以连为如下形式,这样更直观
        # o3 = torch.max(torch.sigmoid(self.fc(x)), dim=-1)[0]
        return o4


class Conv1Model(nn.Module):
    """一维卷积神经网络模型
    包含一个卷积层，kernel_size=7的卷积核
    """
    def __init__(self):
        super(Conv1Model, self).__init__()
        # 定义一维卷积层：输入通道1，输出通道50，卷积核大小7
        self.conv = nn.Conv1d(in_channels=1, out_channels=50, kernel_size=7)

    def forward(self, x):
        """前向传播函数
        Args:
            x: 输入张量，形状为 [批次大小,1,15]
        """
        o1 = self.conv(x)                # 卷积操作
        o2 = torch.max(o1, dim=-1)[0]    # 第一次最大池化
        o3 = torch.max(o2, dim=-1)[0]    # 第二次最大池化
        o4 = torch.sigmoid(o3)           # sigmoid激活
        return o4


class FC2Model(nn.Module):
    """双层全连接神经网络模型
    包含两个线性层，中间使用ReLU激活函数，最后通过sigmoid激活函数输出
    """
    def __init__(self):
        super(FC2Model, self).__init__()
        # 第一个全连接层：输入特征15维，输出特征50维
        self.fc1 = nn.Linear(in_features=15, out_features=50)
        self.relu = nn.ReLU()                    # ReLU激活函数
        # 第二个全连接层：输入特征50维，输出特征1维
        self.fc2 = nn.Linear(in_features=50, out_features=1)

    def forward(self, x):
        """前向传播函数
        Args:
            x: 输入张量，形状为 [批次大小,1,15]
        Returns:
            经过处理后的一维输出张量
        """
        o1 = self.fc1(x)                 # 第1步：第一个线性变换
        o2 = self.relu(o1)               # 第2步：ReLU激活
        o3 = self.fc2(o2)                # 第3步：第二个线性变换
        o4 = torch.sigmoid(o3)           # 第4步：sigmoid激活
        o5 = o4.reshape(-1)              # 第5步：展平成一维张量
        return o5


class MultiConvModel(nn.Module):
    """多层卷积神经网络模型
    包含两个卷积层，一个ReLU激活函数和一个最大池化层
    """
    def __init__(self):
        super(MultiConvModel, self).__init__()
        # 第一个卷积层：输入通道1，输出通道8，卷积核大小3
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3)
        self.relu = nn.ReLU()                    # ReLU激活函数
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)  # 最大池化层
        # 第二个卷积层：输入通道8，输出通道8，卷积核大小3
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3)

    def forward(self, x):
        """前向传播函数
        Args:
            x: 输入张量，形状为 [批次大小,1,15]
        """
        o1 = self.conv1(x)               # 第一次卷积
        o2 = self.relu(o1)               # ReLU激活
        o3 = self.maxpool(o2)            # 最大池化
        o4 = self.conv2(o3)              # 第二次卷积
        o5 = torch.max(o4, dim=-1)[0]    # 第一次全局最大池化
        o6 = torch.max(o5, dim=-1)[0]    # 第二次全局最大池化
        o7 = torch.sigmoid(o6)           # sigmoid激活
        return o7


class ObjFunc:
    """目标函数类
    用于判断输入序列是否满足特定的边缘模式
    """
    feature_length = 7    # 特征长度
    threshold = 0.3      # 判断边缘的阈值

    def _is_edge(self, xs, start, edge1=True):
        """判断是否为边缘
        Args:
            xs: 输入序列
            start: 起始位置
            edge1: True表示上升边缘，False表示下降边缘
        """
        diff = xs[start] - xs[start + 1]
        ret = diff > self.threshold if edge1 else diff < -self.threshold
        return ret

    def _is_feature(self, xs):
        assert len(xs) == self.feature_length

        return self._is_edge(xs, 0) \
            and self._is_edge(xs, 2, edge1=False) \
            and self._is_edge(xs, 3) \
            and self._is_edge(xs, 5, edge1=False)

    def __call__(self, xs):
        """10
        Args:
            xs: a list of double
        return: bool
        """
        assert len(xs) >= self.feature_length

        for i in range(len(xs) - self.feature_length):
            sub_xs = xs[i:i + 7]
            if self._is_feature(sub_xs):
                return True

        return False


class Pattern1DDataset(torch.utils.data.Dataset):
    """一维模式数据集
    生成用于训练和测试的数据集，包含正样本和负样本
    """
    def __init__(self, total_count, length):
        """
        Args:
            total_count: 总样本数
            length: 每个样本的长度
        """
        pos_count = total_count // 2     # 正样本数量
        neg_count = total_count - pos_count  # 负样本数量
        pos_samples = []
        neg_samples = []

        obj_func = ObjFunc()
        # 生成满足条件的正样本和负样本
        while len(pos_samples) < pos_count:
            xs = torch.rand(length).tolist()  # 随机生成序列
            y = obj_func(xs)
            if y:
                pos_samples.append(xs)
            elif len(neg_samples) < neg_count:
                neg_samples.append(xs)

        # 将样本转换为张量格式
        self.data = torch.FloatTensor([pos_samples + neg_samples]).reshape((total_count, 1, length))
        self.labels = torch.FloatTensor([1.0] * pos_count + [0.0] * neg_count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Datasets:
    def __init__(self, batch_size, total_count=1000, length=15):
        self.train_loader = torch.utils.data.DataLoader(Pattern1DDataset(total_count, length), batch_size=batch_size,
                                                        shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(Pattern1DDataset(total_count, length), batch_size=batch_size * 2,
                                                       shuffle=True)


class Trainer:
    """训练器类
    处理模型训练、测试、保存和结果可视化
    """
    def __init__(self, datasets, model, optimizer, loss_fn, results_path='D:\\ai\\models'):
        """初始化训练器
        Args:
            datasets: 包含训练和测试数据加载器的数据集对象
            model: 要训练的神经网络模型
            optimizer: 优化器对象，用于更新模型参数
            loss_fn: 损失函数
            results_path: 模型和训练结果的保存路径
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
        self.model.train()  # 将模型设置为训练模式

        losses = []
        bar = tqdm(self.datasets.train_loader)
        for data, target in bar:
            self.optimizer.zero_grad()    # 清除之前的梯度

            output = self.model(data)     # 前向传播
            loss = self.loss_fn(output, target)  # 计算损失

            loss.backward()               # 反向传播
            self.optimizer.step()         # 更新参数

            bar.set_description(msg_format.format(loss.item()))  # 更新进度条描述
            losses.append(loss.item())
        return losses

    def test(self):
        """在测试集上评估模型
        Returns:
            test_loss: float, 平均测试损失
            accuracy: float, 测试准确率
        """
        self.model.eval()  # 将模型设置为评估模式

        count = len(self.datasets.test_loader.dataset)
        test_loss = 0
        correct = 0
        with torch.no_grad():  # 不计算梯度
            for data, target in self.datasets.test_loader:
                output = self.model(data)
                # 累加批次损失
                test_loss += self.loss_fn(output, target).item() * len(data)
                # 计算正确预测的样本数
                correct += output.gt(0.5).eq(target.gt(0.5)).sum().item()

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
        # 将训练结果保存为CSV文件
        self.train_df = pd.DataFrame(data=all_losses, columns=["train_loss", "val_loss", "accuracy"])
        self.train_df.to_csv(join(self.results_path, "train.csv"), index=False)

    def save_model(self):
        """保存模型参数到文件"""
        torch.save(self.model.state_dict(), join(self.results_path, 'model.pth'))

    def plot(self):
        """绘制训练过程中的损失和准确率曲线"""
        import matplotlib.pyplot as plt
        # 绘制训练损失和验证损失曲线
        self.train_df[["train_loss", "val_loss"]].ffill().plot(title="loss", grid=True, logy=False)
        # 绘制准确率曲线
        self.train_df[["accuracy"]].dropna().plot(title="accuracy", grid=True)
        plt.show()


def train():
    """主训练函数
    设置随机种子，初始化数据集，模型，损失函数和优化器，并开始训练
    """
    torch.manual_seed(0)  # 设置随机种子以确保结果可重现
    datasets = Datasets(100, total_count=10000)  # 初始化数据集

    # 选择要使用的模型（取消注释选择不同的模型）
    # model = FC1Model()
    # model = Conv1Model()
    # model = FC2Model()
    model = MultiConvModel()

    # 设置损失函数和优化器
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 创建训练器并开始训练
    trainer = Trainer(datasets, model=model, optimizer=optimizer,
                      loss_fn=loss_fn, results_path="D:\\ai\\models")
    trainer.train(num_epoch=100)
    print(list(model.parameters()))
    trainer.plot()


if __name__ == "__main__":
    train()
