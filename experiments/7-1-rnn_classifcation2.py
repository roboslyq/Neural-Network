from os.path import join, exists
from os import mkdir
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
"""
RNN 的embedding相关

这段代码实现了一个更复杂的序列分类任务：
1、生成随机的0-9数字序列，找出序列中出现次数最多的数字作为标签
2、提供了两种模型实现：
    RNNClassifier1：直接使用LSTM处理数值输入
    RNNClassifier2：先通过词嵌入层将数字转换为向量，再用LSTM处理
使用负对数似然损失函数和Adam优化器进行训练
3. 提供了完整的训练、测试和可视化功能主要改进：
    支持词嵌入处理离散数字输入
    扩展到10分类问题（之前是3分类）
4. 增加了训练轮数（200个epoch）
5. 增加了数据集大小（5000个样本）
"""

def obj_func(xs):
    """计算序列中出现次数最多的数字
    Args:
        xs: 包含0-9数字的列表
    Returns:
        int: 返回出现频率最高的数字
    """
    val = pd.Series(xs).value_counts().index[0]  # 使用pandas统计最频繁的数字
    return int(val)


class ClassificationDataset(torch.utils.data.Dataset):
    """分类任务数据集
    生成随机的0-9序列，并找出其中出现次数最多的数字作为标签
    """
    def __init__(self, total_count, length, embedding=False):
        """初始化数据集
        Args:
            total_count: 样本总数
            length: 每个序列的长度
            embedding: 是否使用词嵌入模式
                      True: 生成形状为 [total_count, length] 的数据
                      False: 生成形状为 [total_count, length, 1] 的数据
        """
        # 根据embedding参数生成不同形状的随机数据
        self.data = torch.randint(0, 10, (total_count, length)) if embedding else \
                   torch.randint(0, 10, (total_count, length, 1)).type(torch.FloatTensor)
        # 计算每个序列中出现最多的数字作为标签
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
    def __init__(self, batch_size, total_count=1000, length=15, embedding=False):
        """初始化数据加载器
        Args:
            batch_size: 批次大小
            total_count: 数据集样本总数，默认1000
            length: 序列长度，默认15
            embedding: 是否使用词嵌入模式
        """
        # 创建训练和测试数据集加载器
        self.train_loader = torch.utils.data.DataLoader(
            ClassificationDataset(total_count, length, embedding),
            batch_size=batch_size, 
            shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            ClassificationDataset(total_count, length, embedding),
            batch_size=batch_size * 2, 
            shuffle=True
        )


class RNNClassifier1(nn.Module):
    """基础RNN分类器模型
    使用LSTM直接处理数值输入进行分类
    """
    def __init__(self, hidden_size=10):
        """初始化模型
        Args:
            hidden_size: LSTM隐藏层的维度，默认10
        """
        super(RNNClassifier1, self).__init__()
        self.hidden_size = hidden_size
        # LSTM层，输入维度为1，隐藏层维度为hidden_size
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        # 全连接层，将LSTM的输出映射到10个类别（数字0-9）
        self.fc = nn.Linear(in_features=hidden_size, out_features=10)

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


class RNNClassifier2(nn.Module):
    """带词嵌入的RNN分类器模型
    使用词嵌入层将输入数字转换为向量，再用LSTM处理
    """
    def __init__(self, hidden_size=10):
        """初始化模型
        Args:
            hidden_size: LSTM隐藏层的维度，默认10
        """
        super(RNNClassifier2, self).__init__()
        self.hidden_size = hidden_size
        # 词嵌入层，将0-9的数字映射为5维向量
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=5)
        # LSTM层，输入维度为5（词嵌入维度），隐藏层维度为hidden_size
        self.lstm = nn.LSTM(input_size=5, hidden_size=hidden_size, batch_first=True)
        # 全连接层，将LSTM的输出映射到10个类别
        self.fc = nn.Linear(in_features=hidden_size, out_features=10)

    def forward(self, x):
        """前向传播函数
        Args:
            x: 输入张量，形状为 [批次大小,序列长度]
        Returns:
            对数概率分布，形状为 [批次大小,类别数]
        """
        o1 = torch.squeeze(x)  # 去除多余的维度
        o2 = self.embedding(o1)  # 词嵌入
        o3 = self.lstm(o2)  # LSTM处理序列
        o4 = torch.squeeze(o3[1][0])  # 获取最后一个时间步的隐藏状态
        o5 = self.fc(o4)  # 全连接层分类
        o6 = torch.log_softmax(o5, dim=-1)  # 计算对数概率
        return o6


class Trainer:
    """训练器类
    用于管理RNN分类模型的训练、验证、测试过程，并记录训练结果
    
    Attributes:
        datasets: 包含训练集和测试集的数据集对象
        model: 要训练的RNN模型（RNNClassifier1或RNNClassifier2）
        optimizer: 优化器对象，用于更新模型参数
        loss_fn: 损失函数（通常使用NLLLoss）
        results_path: 模型和训练结果的保存路径
        train_df: 用于记录训练过程中的损失值和准确率的DataFrame
    """
    def __init__(self, datasets, model, optimizer, loss_fn, results_path='results'):
        """初始化训练器
        
        Args:
            datasets: 数据集对象，包含训练集和测试集
            model: RNN分类模型
            optimizer: 优化器对象（通常使用Adam）
            loss_fn: 损失函数（通常使用NLLLoss）
            results_path: 结果保存路径，默认为'results'
        """
        self.datasets = datasets
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.results_path = results_path
        self.train_df = None

    def train_epoch(self, msg_format):
        """执行一个训练epoch
        
        Args:
            msg_format: 进度条显示的消息格式字符串
            
        Returns:
            list: 该epoch中所有batch的损失值列表
        """
        self.model.train()  # 设置为训练模式
        losses = []
        
        # 使用tqdm创建进度条
        bar = tqdm(self.datasets.train_loader)
        for data, target in bar:
            # 清空梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            output = self.model(data)
            # 计算损失
            loss = self.loss_fn(output, target)
            
            # 反向传播
            loss.backward()
            # 更新参数
            self.optimizer.step()
            
            # 更新进度条描述
            bar.set_description(msg_format.format(loss.item()))
            # 记录损失值
            losses.append(loss.item())
            
        return losses

    def test(self):
        """执行测试/验证
        
        Returns:
            tuple: (平均测试损失, 测试准确率)
        """
        self.model.eval()  # 设置为评估模式
        
        # 获取测试集大小
        count = len(self.datasets.test_loader.dataset)
        test_loss = 0
        correct = 0
        
        # 关闭梯度计算
        with torch.no_grad():
            for data, target in self.datasets.test_loader:
                # 前向传播
                output = self.model(data)
                # 累加批次损失
                test_loss += self.loss_fn(output, target).item() * len(data)
                # 统计正确预测数
                correct += output.argmax(dim=-1).eq(target).sum().item()

        # 计算平均损失和准确率
        return test_loss / count, correct / count

    def train(self, num_epoch):
        """执行完整的训练过程
        
        Args:
            num_epoch: 训练轮数
            
        训练过程包括：
        1. 在每个epoch中进行训练和验证
        2. 记录训练和验证的损失值与准确率
        3. 保存模型和训练结果
        """
        # 初始测试
        val_loss, accuracy = self.test()
        # 初始化记录列表
        all_losses = [[None, val_loss, accuracy]]

        # 训练循环
        for epoch in range(num_epoch):
            # 训练阶段
            train_losses = self.train_epoch(
                f'train {epoch}/{num_epoch} -- loss: {{:3.2f}}, val_loss: {val_loss:3.2f}, accuracy: {accuracy:.1%}')
            
            # 测试阶段
            val_loss, accuracy = self.test()
            
            # 记录每个batch的训练损失
            all_losses.extend([
                [train_loss, None, None]
                for train_loss in train_losses
            ])
            # 记录验证结果
            all_losses.append([None, val_loss, accuracy])

        # 保存模型和训练结果
        self.save_model()
        self.train_df = pd.DataFrame(data=all_losses, columns=["train_loss", "val_loss", "accuracy"])
        self.train_df.to_csv(join(self.results_path, "train.csv"), index=False)

    def save_model(self):
        """保存训练好的模型
        
        如果结果目录不存在则创建，然后将模型状态字典保存到文件
        """
        if not exists(self.results_path):
            mkdir(self.results_path)
        torch.save(self.model.state_dict(), join(self.results_path, 'model.pth'))

    def plot(self):
        """绘制训练过程中的损失和准确率曲线
        
        生成两个图表：
        1. 训练损失和验证损失的曲线
        2. 准确率曲线
        """
        import matplotlib.pyplot as plt
        # 绘制损失曲线
        self.train_df[["train_loss", "val_loss"]].ffill().plot(
            title="loss", grid=True, logy=False)
        # 绘制准确率曲线
        self.train_df[["accuracy"]].dropna().plot(
            title="accuracy", grid=True)
        plt.show()


def train():
    """主训练函数"""
    torch.manual_seed(0)  # 设置随机种子
    
    # 选择是否使用词嵌入模型
    use_embedding = True
    # 初始化数据集
    datasets = Datasets(100, total_count=5000, length=20, embedding=use_embedding)
    # 根据use_embedding选择使用的模型
    model = RNNClassifier2() if use_embedding else RNNClassifier1()

    # 设置损失函数（负对数似然损失）和优化器
    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    
    # 创建训练器并开始训练
    trainer = Trainer(datasets, model=model, optimizer=optimizer,
                      loss_fn=loss_fn, results_path="results")

    trainer.train(num_epoch=200)  # 训练200个epoch
    trainer.plot()  # 绘制训练结果


if __name__ == "__main__":
    train()
