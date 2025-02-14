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
    用于管理深度学习模型的训练、验证、测试过程，并记录训练结果
    
    Attributes:
        datasets: 包含训练集和测试集的数据集对象
        model: 要训练的神经网络模型
        optimizer: 优化器对象，用于更新模型参数
        loss_fn: 损失函数
        results_path: 模型和训练结果的保存路径
        train_df: 用于记录训练过程中的损失值和准确率的DataFrame
    """
    def __init__(self, datasets, model, optimizer, loss_fn, results_path='D:\\ai\\models'):
        """初始化训练器
        
        Args:
            datasets: 数据集对象，包含训练集和测试集
            model: 要训练的神经网络模型
            optimizer: 优化器对象
            loss_fn: 损失函数
            results_path: 结果保存路径，默认为'D:\\ai\\models'
        """
        self.datasets = datasets
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.results_path = results_path
        self.train_df = None

    def train(self, num_epoch):
        """执行模型训练过程
        
        Args:
            num_epoch: 训练轮数
            
        训练过程包括：
        1. 在每个epoch中进行训练和验证
        2. 记录训练和验证的损失值与准确率
        3. 保存训练结果到DataFrame中
        """
        # 初始化记录训练过程的列表
        epochs, train_losses, train_accuracies = [], [], []
        test_losses, test_accuracies = [], []
        
        # 开始训练循环
        for epoch in tqdm(range(num_epoch)):
            # 训练阶段
            train_loss, train_accuracy = self._train_one_epoch()
            # 测试阶段
            test_loss, test_accuracy = self._test()
            
            # 记录训练过程数据
            epochs.append(epoch)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

        # 将训练结果保存到DataFrame中
        self.train_df = pd.DataFrame({
            'epoch': epochs,
            'train_loss': train_losses,
            'train_accuracy': train_accuracies,
            'test_loss': test_losses,
            'test_accuracy': test_accuracies,
        })

    def _train_one_epoch(self):
        """执行一个训练epoch
        
        Returns:
            tuple: (平均训练损失, 训练准确率)
        """
        self.model.train()  # 设置为训练模式
        total_loss = 0
        correct_count = 0
        
        # 遍历训练数据集
        for x, y in self.datasets.train_loader:
            # 清空梯度
            self.optimizer.zero_grad()
            # 前向传播
            y_pred = self.model(x)
            # 计算损失
            loss = self.loss_fn(y_pred, y)
            # 反向传播
            loss.backward()
            # 更新参数
            self.optimizer.step()
            
            # 累计损失和正确预测数
            total_loss += loss.item()
            correct_count += ((y_pred > 0.5) == y).sum().item()

        # 计算平均损失和准确率
        avg_loss = total_loss / len(self.datasets.train_loader)
        accuracy = correct_count / len(self.datasets.train_loader.dataset)
        return avg_loss, accuracy

    def _test(self):
        """执行测试/验证
        
        Returns:
            tuple: (平均测试损失, 测试准确率)
        """
        self.model.eval()  # 设置为评估模式
        total_loss = 0
        correct_count = 0
        
        # 关闭梯度计算
        with torch.no_grad():
            # 遍历测试数据集
            for x, y in self.datasets.test_loader:
                # 前向传播
                y_pred = self.model(x)
                # 计算损失
                loss = self.loss_fn(y_pred, y)
                
                # 累计损失和正确预测数
                total_loss += loss.item()
                correct_count += ((y_pred > 0.5) == y).sum().item()

        # 计算平均损失和准确率
        avg_loss = total_loss / len(self.datasets.test_loader)
        accuracy = correct_count / len(self.datasets.test_loader.dataset)
        return avg_loss, accuracy

    def plot(self):
        """绘制训练过程中的损失和准确率曲线
        
        生成两个子图：
        1. 训练集和测试集的损失曲线
        2. 训练集和测试集的准确率曲线
        """
        import matplotlib.pyplot as plt
        
        # 创建包含两个子图的图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # 绘制损失曲线
        ax1.plot(self.train_df['epoch'], self.train_df['train_loss'], label='train')
        ax1.plot(self.train_df['epoch'], self.train_df['test_loss'], label='test')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax1.legend()
        
        # 绘制准确率曲线
        ax2.plot(self.train_df['epoch'], self.train_df['train_accuracy'], label='train')
        ax2.plot(self.train_df['epoch'], self.train_df['test_accuracy'], label='test')
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('accuracy')
        ax2.legend()
        
        # 保存图表
        plt.savefig(join(self.results_path, 'train_process.png'))


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
