from os.path import join, exists
from os import mkdir

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import CoNLL2000Chunking

from tqdm import tqdm
import pandas as pd

"""
序列标注实现
目标：使用LSTM/BiLSTM实现词性标注任务（CoNLL2000数据集）

主要特点：
1. 支持单向LSTM和双向LSTM两种模型
2. 处理变长序列和填充
3. 使用词嵌入表示输入
4. 实现了带掩码的损失计算
"""

# 加载CoNLL2000数据集
train_iter, test_iter = CoNLL2000Chunking(root='datasets', split=('train', 'test'))
# 提取词和标签对
train_set = [(words, tags) for words, _, tags in train_iter]
test_set = [(words, tags) for words, _, tags in test_iter]

# 构建词表
words_series = pd.Series([t for words, tags in train_set for t in words])
word_counts = words_series.value_counts()
vocab_size = len(word_counts)
print("vocab size:", vocab_size)

# 词表裁剪：只保留出现次数>=5的词
MIN_WORD_COUNT = 5
truncated_word_counts = word_counts[word_counts >= MIN_WORD_COUNT]
truncated_vocab_size = len(truncated_word_counts)
print("truncated vocab size:", truncated_vocab_size)
print(f"Retention: {truncated_vocab_size/vocab_size:.1%}")
vocab = sorted(truncated_word_counts.index)

# 添加特殊token
PADDING_IDX = 0
vocab.insert(PADDING_IDX, "<padding>")  # 用于填充序列

UNKNOWN_IDX = 1
vocab.insert(UNKNOWN_IDX, "<unknown>")  # 用于处理未知词

# 创建词到索引的映射
token2idx = {token: idx for idx, token in enumerate(vocab)}

# 构建标签集合并创建映射
tags = sorted(list({t for words, tags in train_set for t in tags}))
tag2idx = {tag: idx for idx, tag in enumerate(tags)}


class ChunkingDataset(Dataset):
    """序列标注数据集类"""
    def __init__(self, dataset, max_length):
        """初始化数据集
        Args:
            dataset: 原始数据集
            max_length: 序列最大长度
        """
        xs = []
        ys = []
        for words, tags in dataset:
            xs.append(self.nodes2index(words, token2idx, max_length))
            ys.append(self.nodes2index(tags, tag2idx, max_length))

        self.data = torch.LongTensor(xs)
        self.labels = torch.LongTensor(ys)

    @staticmethod
    def nodes2index(nodes, node2idx, max_length):
        """将token序列转换为索引序列
        Args:
            nodes: token序列
            node2idx: token到索引的映射
            max_length: 最大长度
        Returns:
            填充到固定长度的索引序列
        """
        # 将token转换为索引
        indexes = [node2idx.get(t, UNKNOWN_IDX) for t in nodes]
        # 截断或填充到指定长度
        indexes = indexes[:max_length]
        indexes += [PADDING_IDX] * (max_length - len(indexes))
        return indexes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Datasets:
    def __init__(self, batch_size, max_length=100):
        self.train_loader = DataLoader(ChunkingDataset(train_set, max_length),
                                       batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(ChunkingDataset(test_set, max_length),
                                      batch_size=batch_size * 2, shuffle=True)


class LstmTagger(nn.Module):
    """单向LSTM序列标注模型"""
    def __init__(self, num_classes, num_embeddings, embedding_dim, hidden_size):
        """初始化模型
        Args:
            num_classes: 标签类别数
            num_embeddings: 词表大小
            embedding_dim: 词嵌入维度
            hidden_size: LSTM隐藏层维度
        """
        super(LstmTagger, self).__init__()

        self.num_classes = num_classes
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # 词嵌入层
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        # LSTM层
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        # 输出层
        self.fc = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, x):
        """前向传播
        Args:
            x: 输入序列，形状为 [批次大小,序列长度]
        Returns:
            每个位置的标签概率分布
        """
        o1 = self.embedding(x)  # 词嵌入
        o2 = self.lstm(o1)      # LSTM处理
        o3 = self.fc(o2[0])     # 全连接层
        o4 = torch.log_softmax(o3, dim=-1)  # 计算对数概率
        return o4


class BiLSTMTagger(nn.Module):
    """双向LSTM序列标注模型"""
    def __init__(self, num_classes, num_embeddings, embedding_dim, hidden_size=256):
        """初始化模型
        Args:
            num_classes: 标签类别数
            num_embeddings: 词表大小
            embedding_dim: 词嵌入维度
            hidden_size: LSTM隐藏层维度
        """
        super(BiLSTMTagger, self).__init__()

        self.num_classes = num_classes
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # 词嵌入层
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        # 双向LSTM层
        o2 = hidden_size * 2  # 双向LSTM的输出维度是隐藏层维度的两倍
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, 
                           batch_first=True, bidirectional=True)
        # 输出层
        self.fc = nn.Linear(in_features=o2, out_features=num_classes)

    def forward(self, x):
        """前向传播
        Args:
            x: 输入序列，形状为 [批次大小,序列长度]
        Returns:
            每个位置的标签概率分布
        """
        o1 = self.embedding(x)  # 词嵌入
        o3 = self.lstm(o1)      # 双向LSTM处理
        o4 = self.fc(o3[0])     # 全连接层
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
        """训练一个epoch"""
        self.model.train()
        losses = []
        bar = tqdm(self.datasets.train_loader)
        for tokens, target in bar:
            self.optimizer.zero_grad()

            _tokens = tokens.to(self.device)
            masks = _tokens.gt(0.0)  # 创建掩码，标记非填充位置
            output = self.model(_tokens)
            # 计算带掩码的损失
            batch_losses = self.loss(output, target.to(self.device), masks)
            loss = batch_losses / masks.sum()

            loss.backward()
            self.optimizer.step()

            bar.set_description(msg_format.format(loss.item()))
            losses.append(loss.item())
        return losses

    def test(self):
        self.model.eval()

        tokens_count = 0
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for tokens, target in self.datasets.test_loader:
                _tokens = tokens.to(self.device)
                _target = target.to(self.device)

                masks = _tokens.gt(0.0)
                tokens_count += masks.sum()

                output = self.model(_tokens)
                test_loss += self.loss(output, _target, masks)

                correct += (output.argmax(dim=-1).eq(_target) * masks).sum()

        return (test_loss / tokens_count).item(), (correct / tokens_count).item()

    def loss(self, output, target, masks):
        """计算带掩码的损失
        只计算非填充位置的损失
        """
        masked_losses = self.loss_fn(output.permute(dims=(0, 2, 1)), target) * masks
        return masked_losses.sum()

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
    datasets = Datasets(300, max_length=60)  # 创建数据集

    # 创建模型（可以选择LstmTagger或BiLSTMTagger）
    model = LstmTagger(len(tags), num_embeddings=len(vocab), embedding_dim=50, hidden_size=50)
    # model = BiLSTMTagger(len(tags), num_embeddings=len(vocab), embedding_dim=50, hidden_size=50)

    # 创建训练组件
    loss_fn = torch.nn.NLLLoss(reduction="none")  # 使用不带reduction的损失函数，以支持掩码
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.0006)
    
    # 创建训练器并开始训练
    trainer = Trainer(datasets, model=model, optimizer=optimizer,
                      loss_fn=loss_fn, results_path="results")

    trainer.train(num_epoch=50)  # 训练50个epoch
    trainer.plot()  # 绘制训练结果


if __name__ == "__main__":
    train()
