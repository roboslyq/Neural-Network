from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import pandas as pd
"""
RNN语言模型实现
目标：使用LSTM实现字符级语言模型，可以生成莎士比亚风格的文本

主要特点：
1. 字符级建模，直接学习字符序列的概率分布
2. 使用LSTM网络捕获长期依赖
3. 支持温度参数控制生成文本的多样性
4. 实现了文本生成的采样策略
"""

# 加载莎士比亚文本数据
with open("datasets/shakespeare.txt", encoding="utf8") as f:
    text = f.read()

# 划分训练集和验证集（7:3）
train_size = len(text) * 7 // 10
train_set = text[:train_size]
val_set = text[train_size:]

# 构建字符级词表
vocab = sorted(set(train_set))
vocab_size = len(vocab)
print("vocab size:", vocab_size)

# 创建字符到索引的映射
char2idx = {char: idx for idx, char in enumerate(vocab)}


class LMDataset(Dataset):
    """语言模型数据集类
    将文本切分为固定长度的序列，每个序列对应的标签是其后移一位的序列
    """
    def __init__(self, dataset, max_length):
        """初始化数据集
        Args:
            dataset: 原始文本
            max_length: 序列长度
        """
        xs = []
        ys = []
        # 按固定长度切分文本，步长为max_length
        for i in range(0, len(dataset) - max_length - 2, max_length):
            txt = dataset[i : i + max_length + 1]
            # 将字符转换为索引
            indices = [char2idx[char] for char in txt]
            # 输入序列和目标序列错开一位
            xs.append(indices[:-1])  # 输入：当前字符
            ys.append(indices[1:])   # 目标：下一个字符

        self.xs = torch.LongTensor(xs)
        self.ys = torch.LongTensor(ys)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


class Datasets:
    def __init__(self, batch_size, max_length=100):
        self.train_loader = DataLoader(
            LMDataset(train_set, max_length), batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            LMDataset(val_set, max_length), batch_size=batch_size * 2, shuffle=True
        )


class LstmLM(nn.Module):
    """LSTM语言模型
    使用LSTM网络学习字符序列的概率分布
    """
    def __init__(self, num_classes, num_embeddings, embedding_dim, hidden_size):
        """初始化模型
        Args:
            num_classes: 字符类别数
            num_embeddings: 词表大小
            embedding_dim: 字符嵌入维度
            hidden_size: LSTM隐藏层维度
        """
        super(LstmLM, self).__init__()

        self.num_classes = num_classes
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # 字符嵌入层
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        # LSTM层
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        # 输出层
        self.fc = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, x):
        """前向传播（训练模式）
        Args:
            x: 输入序列，形状为 [批次大小,序列长度]
        Returns:
            每个位置的字符概率分布
        """
        o1 = self.embedding(x)  # 字符嵌入
        o2 = self.lstm(o1)      # LSTM处理
        o3 = self.fc(o2[0])     # 全连接层
        o4 = torch.log_softmax(o3, dim=-1)  # 计算对数概率
        return o4

    def predict(self, x, states=None, temperature=1.0):
        """生成预测（推理模式）
        Args:
            x: 输入序列
            states: LSTM隐藏状态
            temperature: 采样温度，控制生成的随机性
        Returns:
            下一个字符的概率分布和新的隐藏状态
        """
        o1 = self.embedding(x)  # 字符嵌入
        o2 = self.lstm(o1, states)  # LSTM处理，使用上一次的状态
        o3 = self.fc(o2[0])     # 全连接层
        if temperature != 1.0:
            o3 /= temperature    # 应用温度缩放
        o4 = torch.softmax(o3, dim=-1)  # 计算概率分布
        return o4, o2[1]  # 返回概率分布和新的状态


class Trainer:
    def __init__(self, datasets, model, optimizer, loss_fn, results_path="results"):
        self.datasets = datasets
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.results_path = Path(results_path)

        self.train_df = None

        # device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using device: ", self.device)
        model.to(self.device)

    def train_epoch(self, msg_format):
        self.model.train()

        losses = []
        progress_bar = tqdm(self.datasets.train_loader)
        for tokens, target in progress_bar:
            self.optimizer.zero_grad()

            _tokens = tokens.to(self.device)
            masks = _tokens.gt(0.0)
            output = self.model(_tokens)
            batch_losses = self.loss(output, target.to(self.device), masks)
            loss = batch_losses / masks.sum()

            loss.backward()
            self.optimizer.step()

            progress_bar.set_description(msg_format.format(loss.item()))

            losses.append(loss.item())
        return losses

    def validate(self):
        self.model.eval()

        tokens_count = 0
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for tokens, target in self.datasets.val_loader:
                _tokens = tokens.to(self.device)
                _target = target.to(self.device)

                masks = _tokens.gt(0.0)
                tokens_count += masks.sum()

                output = self.model(_tokens)
                val_loss += self.loss(output, _target, masks)

                correct += (output.argmax(dim=-1).eq(_target) * masks).sum()

        return (val_loss / tokens_count).item(), (correct / tokens_count).item()

    def loss(self, output, target, masks):
        masked_losses = self.loss_fn(output.permute(dims=(0, 2, 1)), target) * masks
        return masked_losses.sum()

    def train(self, num_epoch):
        val_loss, accuracy = self.validate()
        all_losses = [[None, val_loss, accuracy]]

        for epoch in range(num_epoch):
            # train
            train_losses = self.train_epoch(
                f"train {epoch}/{num_epoch} -- loss: {{:3.2f}}, val_loss: {val_loss:3.2f}, accuracy: {accuracy:.1%}"
            )

            # validate
            val_loss, accuracy = self.validate()
            all_losses.extend([[train_loss, None, None] for train_loss in train_losses])
            all_losses.append([None, val_loss, accuracy])
        print(f"final accuracy: {accuracy}")

        self.save_model()
        self.train_df = pd.DataFrame(
            data=all_losses, columns=["train_loss", "val_loss", "accuracy"]
        )
        self.train_df.to_csv(self.results_path / "train.csv", index=False)

    def save_model(self):
        self.results_path.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), self.results_path / "model.pth")

        with open(self.results_path / "vocab.json", "w", encoding="utf8") as f:
            f.write(json.dumps(vocab))

    def plot(self):
        import matplotlib.pyplot as plt

        self.train_df[["train_loss", "val_loss"]].ffill().plot(
            title="loss", grid=True, logy=False
        )
        self.train_df[["accuracy"]].dropna().plot(title="accuracy", grid=True)
        plt.show()


def train():
    """主训练函数"""
    torch.manual_seed(0)  # 设置随机种子
    datasets = Datasets(300, max_length=60)  # 创建数据集

    # 创建模型和训练组件
    model = LstmLM(len(vocab), num_embeddings=len(vocab), 
                   embedding_dim=50, hidden_size=50)
    loss_fn = torch.nn.NLLLoss(reduction="none")  # 使用不带reduction的损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.0006)
    
    # 创建训练器并开始训练
    trainer = Trainer(datasets, model=model, optimizer=optimizer,
                     loss_fn=loss_fn, results_path="results")

    trainer.train(num_epoch=50)  # 训练50个epoch
    trainer.plot()  # 绘制训练结果


class Predictor:
    """文本生成器类"""
    def __init__(self, model_dir):
        """初始化生成器
        Args:
            model_dir: 模型保存目录
        """
        # 加载词表
        with open(model_dir / "vocab.json", "r", encoding="utf8") as f:
            vocab = json.load(f)

        # 创建模型
        model = LstmLM(len(vocab), num_embeddings=len(vocab), 
                      embedding_dim=50, hidden_size=50)
        # 加载预训练参数
        model.load_state_dict(torch.load(model_dir / "model.pth"))
        self.model = model.to(torch.device("cpu"))

    def predict(self, text, max_length=100, temperature=1.0):
        """生成文本
        Args:
            text: 起始文本
            max_length: 生成的最大长度
            temperature: 采样温度
        Returns:
            生成的文本序列
        """
        # 将起始文本转换为索引序列
        tokens = torch.LongTensor([char2idx[char] for char in text])
        generated = ""

        self.model.eval()
        with torch.no_grad():
            states = None
            # 逐字符生成
            for _ in range(max_length):
                # 获取下一个字符的概率分布
                output, states = self.model.predict(tokens, states, 
                                                  temperature=temperature)
                # 使用多项分布采样下一个字符
                predicted_idx = output[-1].multinomial(1).item()
                predicted_char = vocab[predicted_idx]
                generated += predicted_char

                # 更新输入序列
                tokens = torch.LongTensor([predicted_idx])

        return generated


def predict():
    """测试文本生成"""
    predictor = Predictor(Path("results"))
    # 使用换行符作为起始字符，生成100个字符
    text = predictor.predict("\n", max_length=100, temperature=0.5)
    print(text)


if __name__ == "__main__":
    train()
    # predict()
