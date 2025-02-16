from torchtext.datasets import IMDB
from os.path import join, exists
from os import mkdir
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
"""
RNN文本分类实现
目标：使用LSTM网络对IMDB电影评论进行情感分类（正面/负面）

主要特点：
1. 使用词嵌入层将文本转换为向量
2. 使用LSTM处理变长序列
3. 基于最后一个时间步的隐藏状态进行分类
"""

# 加载IMDB数据集
print("loading dataset...")
train_iter, test_iter = IMDB("datasets", split=('train', 'test'))


def tokenize(text):
    """文本分词函数
    将文本转换为小写并按空格分割
    Args:
        text: 输入文本
    Returns:
        分词后的token列表
    """
    return [t.lower() for t in text.split()]


# 对训练集和测试集进行分词处理
train_set = [(label, tokenize(line)) for label, line in tqdm(train_iter, desc="tokenizing trainset...")]
test_set = [(label, tokenize(line)) for label, line in tqdm(test_iter, desc="tokenizing testset...")]

# 构建词表
vocab = sorted(list(set(t for (_, tokens) in train_set for t in tokens)))

# 添加特殊token
PADDING_IDX = 0
vocab.insert(PADDING_IDX, "<padding>")  # 用于填充序列

UNKNOWN_IDX = 1
vocab.insert(UNKNOWN_IDX, "<unknown>")  # 用于处理未知词

# 创建词到索引的映射
token2idx = {token: idx for idx, token in enumerate(vocab)}
# 创建标签到索引的映射
labels = ["neg", "pos"]
label2idx = {label: idx for idx, label in enumerate(labels)}


class IMDBDataset(torch.utils.data.Dataset):
    """IMDB数据集类
    将文本转换为定长的索引序列
    """
    def __init__(self, dataset, max_length):
        """初始化数据集
        Args:
            dataset: 原始数据集
            max_length: 序列最大长度
        """
        xs = []
        ys = []
        for label, tokens in dataset:
            # 将token转换为索引
            indexes = [token2idx.get(t, UNKNOWN_IDX) for t in tokens]
            # 截断或填充到指定长度
            indexes = indexes[:max_length]
            indexes += [PADDING_IDX] * (max_length - len(indexes))
            xs.append(indexes)
            ys.append(label2idx[label])

        self.data = torch.LongTensor(xs)
        self.labels = torch.LongTensor(ys)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Datasets:
    def __init__(self, batch_size, max_length=100):
        self.train_loader = torch.utils.data.DataLoader(IMDBDataset(train_set, max_length),
                                                        batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(IMDBDataset(test_set, max_length),
                                                       batch_size=batch_size * 2, shuffle=True)


class TextClassifier(nn.Module):
    """文本分类模型
    使用词嵌入+LSTM+全连接层进行分类
    """
    def __init__(self, num_embeddings=5000, embedding_dim=50, hidden_size=50):
        """初始化模型
        Args:
            num_embeddings: 词表大小
            embedding_dim: 词嵌入维度
            hidden_size: LSTM隐藏层维度
        """
        super(TextClassifier, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # 词嵌入层
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        # LSTM层
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        # 全连接分类层
        self.fc = nn.Linear(in_features=hidden_size, out_features=2)

    def forward(self, x):
        """前向传播
        Args:
            x: 输入序列，形状为 [批次大小,序列长度]
        Returns:
            分类的对数概率
        """
        o1 = self.embedding(x)  # 词嵌入
        o2 = self.lstm(o1)      # LSTM处理
        o3 = torch.squeeze(o2[1][0])  # 获取最后一个时间步的隐藏状态
        o4 = self.fc(o3)        # 全连接分类
        o5 = torch.log_softmax(o4, dim=-1)  # 计算对数概率
        return o5


class Trainer:
    """训练器类：处理模型训练、测试、保存和可视化"""
    def __init__(self, datasets, model, optimizer, loss_fn, results_path='results'):
        """初始化训练器"""
        self.datasets = datasets
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.results_path = results_path
        self.train_df = None

        # 设置设备(GPU/CPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using device: ", self.device)
        model.to(self.device)

    def train_epoch(self, msg_format):
        self.model.train()

        losses = []
        bar = tqdm(self.datasets.train_loader)
        for data, target in bar:
            self.optimizer.zero_grad()

            output = self.model(data.to(self.device))
            loss = self.loss_fn(output, target.to(self.device))

            loss.backward()
            self.optimizer.step()

            bar.set_description(msg_format.format(loss.item()))

            losses.append(loss.item())
        return losses

    def test(self):
        self.model.eval()

        count = len(self.datasets.test_loader.dataset)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.datasets.test_loader:
                _target = target.to(self.device)
                output = self.model(data.to(self.device))
                test_loss += self.loss_fn(output, _target).item() * len(data)
                correct += output.argmax(dim=-1).eq(_target).sum().item()

        return test_loss / count, correct / count

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
        print(f"final accuracy: {accuracy}")

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
    
    # 创建数据集
    datasets = Datasets(300, max_length=150)

    # 创建模型和训练组件
    model = TextClassifier(num_embeddings=len(vocab), embedding_dim=50, hidden_size=50)
    loss_fn = torch.nn.NLLLoss()  # 使用负对数似然损失
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    
    # 创建训练器并开始训练
    trainer = Trainer(datasets, model=model, optimizer=optimizer,
                      loss_fn=loss_fn, results_path="results")

    trainer.train(num_epoch=30)  # 训练30个epoch
    trainer.plot()  # 绘制训练结果


if __name__ == "__main__":
    train()
