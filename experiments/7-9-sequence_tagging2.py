from os.path import join, exists
from os import mkdir

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import CoNLL2000Chunking

from tqdm import tqdm
import pandas as pd
from bi_lstm_crf import CRF
"""
BI-LSTM-CRF序列标注实现
目标：使用BiLSTM-CRF模型实现序列标注任务（CoNLL2000数据集）

主要特点：
1. 使用双向LSTM捕获上下文信息
2. 使用CRF层建模标签间的依赖关系
3. 支持变长序列和填充处理
4. 实现了带掩码的损失计算
"""

# 加载CoNLL2000数据集
train_iter, test_iter = CoNLL2000Chunking(root='datasets', split=('train', 'test'))
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

# 创建映射字典
token2idx = {token: idx for idx, token in enumerate(vocab)}
tags = sorted(list({t for words, tags in train_set for t in tags}))
tag2idx = {tag: idx for idx, tag in enumerate(tags)}


class ChunkingDataset(Dataset):
    def __init__(self, dataset, max_length):
        xs = []
        ys = []
        for words, tags in dataset:
            xs.append(self.nodes2index(words, token2idx, max_length))
            ys.append(self.nodes2index(tags, tag2idx, max_length))

        self.data = torch.LongTensor(xs)
        self.labels = torch.LongTensor(ys)

    @staticmethod
    def nodes2index(nodes, node2idx, max_length):
        # convert nodes to indexes
        indexes = [node2idx.get(t, UNKNOWN_IDX) for t in nodes]
        # truncate
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


class BiLstmCrfTagger(nn.Module):
    """BiLSTM-CRF序列标注模型
    结合了双向LSTM和条件随机场(CRF)的序列标注模型
    """
    def __init__(self, num_classes, num_embeddings, embedding_dim, hidden_size=256):
        """初始化模型
        Args:
            num_classes: 标签类别数
            num_embeddings: 词表大小
            embedding_dim: 词嵌入维度
            hidden_size: LSTM隐藏层维度
        """
        super(BiLstmCrfTagger, self).__init__()

        self.num_classes = num_classes
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # 词嵌入层
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        # 双向LSTM层
        o2 = hidden_size * 2  # 双向LSTM输出维度是隐藏层维度的两倍
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, 
                           batch_first=True, bidirectional=True)
        # CRF层
        self.crf = CRF(o2, num_classes)

    def features(self, xs):
        """提取特征
        使用BiLSTM提取序列特征
        Args:
            xs: 输入序列
        Returns:
            BiLSTM的输出特征
        """
        o1 = self.embedding(xs)
        return self.lstm(o1)[0]

    def forward(self, xs):
        """前向传播
        Args:
            xs: 输入序列
        Returns:
            预测的标签序列和得分
        """
        features = self.features(xs)
        # 使用CRF进行解码
        scores, tag_seq = self.crf(features, xs.gt(0))
        return tag_seq

    def loss(self, xs, tags):
        """计算损失
        Args:
            xs: 输入序列
            tags: 目标标签序列
        Returns:
            CRF损失值
        """
        features = self.features(xs)
        # 计算CRF损失
        loss = self.crf.loss(features, tags, xs.gt(0))
        return loss


class Trainer:
    """训练器类：处理模型训练、测试、保存和可视化"""
    def __init__(self, datasets, model, optimizer, results_path='results'):
        """初始化训练器"""
        self.datasets = datasets
        self.model = model
        self.optimizer = optimizer
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
            _target = target.to(self.device)
            # 计算CRF损失
            loss = self.model.loss(_tokens, _target)

            loss.backward()
            self.optimizer.step()

            bar.set_description(msg_format.format(loss.item()))
            losses.append(loss.item())
        return losses

    def test(self):
        """测试模型性能"""
        self.model.eval()

        tokens_count = 0
        test_loss = 0
        correct = 0
        count = len(self.datasets.test_loader.dataset)
        
        with torch.no_grad():
            for tokens, target in self.datasets.test_loader:
                _tokens = tokens.to(self.device)
                _target = target.to(self.device)

                masks = _tokens.gt(0.0)
                tokens_count += masks.sum()

                # 获取预测序列
                output = self.model(_tokens)
                # 计算损失
                test_loss += self.model.loss(_tokens, _target) * len(tokens)
                # 计算正确预测的数量
                correct += sum([p == t for ps, ts in zip(output, target.tolist()) 
                              for p, t in zip(ps, ts)])

        return (test_loss / count).item(), (correct / tokens_count).item()

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

    # 创建BiLSTM-CRF模型
    model = BiLstmCrfTagger(len(tags), num_embeddings=len(vocab), 
                           embedding_dim=50, hidden_size=50)

    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.0006)
    
    # 创建训练器并开始训练
    trainer = Trainer(datasets, model=model, optimizer=optimizer, 
                     results_path="results")

    trainer.train(num_epoch=50)  # 训练50个epoch
    trainer.plot()  # 绘制训练结果


if __name__ == "__main__":
    train()
