from torchtext.datasets import IMDB
from os.path import join, exists
from os import mkdir
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
from torchtext.data import get_tokenizer
"""
RNN文本分类实现2 - 改进版本
目标：使用更高级的分词方法和词表裁剪策略改进IMDB情感分类

主要改进：
1. 使用spaCy进行更准确的分词
2. 基于词频进行词表裁剪，减少低频词
3. 添加了GetIndex模块处理序列长度
"""
print("loading dataset...")
train_iter, test_iter = IMDB("datasets", split=('train', 'test'))

# 分词设置
USING_SPACY = True
# 使用spaCy分词器或简单的空格分割
tokenizer = get_tokenizer("spacy", "en_core_web_sm") if USING_SPACY else lambda x: x.split()


def tokenize(text):
    """文本分词函数
    使用spaCy或简单分词，并转换为小写
    Args:
        text: 输入文本
    Returns:
        分词后的token列表
    """
    return [t.lower() for t in tokenizer(text)]


# 对训练集和测试集进行分词
train_set = [(label, tokenize(line)) for label, line in tqdm(train_iter, desc="tokenizing trainset...")]
test_set = [(label, tokenize(line)) for label, line in tqdm(test_iter, desc="tokenizing testset...")]

# 获取最大序列长度
MAX_SEQ_LENGTH = max([len(tokens) for (_, tokens) in train_set])
print("max sequence length:", MAX_SEQ_LENGTH)

# 词表构建和裁剪
words_series = pd.Series([t for (_, tokens) in train_set for t in tokens])
word_counts = words_series.value_counts()
vocab_size = len(word_counts)
print("vocab size:", vocab_size)

# 只保留出现次数大于MIN_WORD_COUNT的词
MIN_WORD_COUNT = 50
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
labels = ["neg", "pos"]
label2idx = {label: idx for idx, label in enumerate(labels)}


class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_length):
        xs = []
        ys = []
        for label, tokens in dataset:
            # convert tokens to indexes
            indexes = [token2idx.get(t, UNKNOWN_IDX) for t in tokens]
            # truncate
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


class GetIndex(nn.Module):
    """序列长度处理模块
    用于获取序列中最后一个非填充位置的索引
    """
    def __init__(self):
        super(GetIndex, self).__init__()

    def forward(self, x):
        """前向传播
        Args:
            x: 输入序列，形状为 [批次大小,序列长度]
        Returns:
            最后一个非填充位置的索引
        """
        o1 = torch.gt(x, other=0.0)  # 找出非填充位置(>0)
        o2 = torch.sum(o1, dim=1, keepdim=True)  # 计算非填充token的数量
        o3 = torch.sub(o2, other=1.0)  # 减1得到最后位置的索引
        o4 = o3.long()  # 转换为整数类型
        o5 = torch.unsqueeze(o4, dim=-1)  # 增加维度
        o6 = o5.expand(-1, -1, 50)  # 扩展维度以匹配隐藏状态
        return o6


class TextClassifier(nn.Module):
    """改进的文本分类模型
    使用GetIndex模块处理变长序列
    """
    def __init__(self, embedding_dim, num_embeddings, hidden_size):
        """初始化模型
        Args:
            embedding_dim: 词嵌入维度
            num_embeddings: 词表大小
            hidden_size: LSTM隐藏层维度
        """
        super(TextClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size

        # 词嵌入层
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=hidden_size)
        # 序列长度处理模块
        self.getindex = GetIndex()
        # LSTM层
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        # 分类层
        self.fc = nn.Linear(in_features=hidden_size, out_features=2)

    def forward(self, x):
        """前向传播
        Args:
            x: 输入序列，形状为 [批次大小,序列长度]
        Returns:
            分类的对数概率
        """
        o1 = self.embedding(x)  # 词嵌入
        o2 = self.getindex(x)   # 获取序列实际长度
        o3 = self.lstm(o1)      # LSTM处理
        # 根据实际长度获取对应位置的隐藏状态
        o4 = torch.gather(o3[0], index=o2, dim=1)  
        o5 = torch.squeeze(o4)  # 压缩维度
        o6 = self.fc(o5)        # 全连接分类
        o7 = torch.log_softmax(o6, dim=-1)  # 计算对数概率
        return o7


class Trainer:
    def __init__(self, datasets, model, optimizer, loss_fn, results_path='results'):
        self.datasets = datasets
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.results_path = results_path

        self.train_df = None

        # device
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
    datasets = Datasets(300, max_length=400)  # 创建数据集

    # 创建模型和训练组件
    model = TextClassifier(num_embeddings=len(vocab), embedding_dim=50, hidden_size=50)
    loss_fn = torch.nn.NLLLoss()  # 使用负对数似然损失
    # 添加权重衰减以防止过拟合
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.0004)
    
    # 创建训练器并开始训练
    trainer = Trainer(datasets, model=model, optimizer=optimizer,
                      loss_fn=loss_fn, results_path="results")

    trainer.train(num_epoch=30)  # 训练30个epoch
    trainer.plot()  # 绘制训练结果


if __name__ == "__main__":
    train()
