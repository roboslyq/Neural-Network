"""
简单聊天模型 - 详细注释版
目标：实现一个简单的问答式聊天模型

基本概念：
- Seq2Seq: 序列到序列模型，用于将输入序列转换为输出序列
- Encoder: 将输入文本转换为向量表示
- Decoder: 将向量表示转换为回复文本
- Attention: 注意力机制，帮助模型关注重要信息
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import torch.nn.functional as F

# 定义特殊标记
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SOS_TOKEN = '<sos>'  # 句子开始
EOS_TOKEN = '<eos>'  # 句子结束

class Vocabulary:
    """词汇表类：处理文本和索引的转换"""
    def __init__(self):
        # 初始化词汇表，添加特殊标记
        self.word2idx = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
            SOS_TOKEN: 2,
            EOS_TOKEN: 3
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        
    def add_word(self, word):
        """添加新词到词汇表"""
        if word not in self.word2idx:
            self.word2idx[word] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = word
            
    def __len__(self):
        return len(self.word2idx)

class ChatDataset(Dataset):
    """聊天数据集类"""
    def __init__(self, questions, answers, vocab):
        self.questions = questions  # 问题列表
        self.answers = answers      # 回答列表
        self.vocab = vocab         # 词汇表

    def __getitem__(self, index):
        # 获取问题和回答的词索引序列
        question = self.text_to_indices(self.questions[index])
        answer = self.text_to_indices(self.answers[index])
        return torch.tensor(question), torch.tensor(answer)

    def text_to_indices(self, text):
        """将文本转换为索引序列"""
        words = text.split()
        return [self.vocab.word2idx.get(word, self.vocab.word2idx[UNK_TOKEN]) 
                for word in words]

    def __len__(self):
        return len(self.questions)

class ChatModel(nn.Module):
    """聊天模型：使用Seq2Seq架构"""
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=256):
        super(ChatModel, self).__init__()
        
        self.hidden_size = hidden_size
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 编码器
        self.encoder = nn.LSTM(
            embedding_dim, 
            hidden_size,
            batch_first=True
        )
        
        # 解码器
        self.decoder = nn.LSTM(
            embedding_dim,
            hidden_size,
            batch_first=True
        )
        
        # 输出层
        self.out = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """前向传播
        Args:
            src: 输入序列（问题）[batch_size, seq_len]
            tgt: 目标序列（回答）[batch_size, seq_len]
            teacher_forcing_ratio: 使用教师强制的概率
        """
        batch_size = src.shape[0]
        max_len = tgt.shape[1]
        vocab_size = self.out.out_features
        
        # 初始化输出张量
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(src.device)
        
        # 编码器前向传播
        # [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(src)
        
        # LSTM期望输入为[batch_size, seq_len, input_size]
        encoder_output, (hidden, cell) = self.encoder(embedded)
        
        # 解码器输入初始化为SOS标记
        decoder_input = torch.tensor([[2]]*batch_size).to(src.device)  # [batch_size, 1]
        
        # 逐步解码
        for t in range(max_len):
            # [batch_size, 1] -> [batch_size, 1, embedding_dim]
            embedded_decoder = self.embedding(decoder_input)
            
            # 确保维度正确
            if embedded_decoder.dim() == 2:
                embedded_decoder = embedded_decoder.unsqueeze(1)
                
            # LSTM解码
            decoder_output, (hidden, cell) = self.decoder(
                embedded_decoder,
                (hidden, cell)
            )
            
            # [batch_size, hidden_size] -> [batch_size, vocab_size]
            prediction = self.out(decoder_output.squeeze(1))
            outputs[:, t] = prediction
            
            # 教师强制：使用真实目标作为下一步输入
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            decoder_input = tgt[:, t] if teacher_force else prediction.argmax(1)
            
        return outputs

class Trainer:
    """训练器类"""
    def __init__(self, model, vocab, device):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.model.to(device)
        
    def train(self, train_loader, num_epochs, learning_rate=0.001):
        """训练模型"""
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD_TOKEN
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, (questions, answers) in enumerate(tqdm(train_loader)):
                questions = questions.to(self.device)
                answers = answers.to(self.device)
                
                # 前向传播
                outputs = self.model(questions, answers)
                
                # 计算损失
                loss = criterion(
                    outputs.view(-1, len(self.vocab)),
                    answers.view(-1)
                )
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch: {epoch+1}, Loss: {avg_loss:.4f}')

    def generate_response(self, question, max_length=20):  # 减小max_length
        """生成回复
        Args:
            question: 输入的问题文本
            max_length: 生成回复的最大长度，默认20个词（避免过长重复）
            
        Returns:
            string: 生成的回复文本
        """
        self.model.eval()
        
        # 将问题转换为索引序列
        question_indices = [self.vocab.word2idx.get(word, self.vocab.word2idx[UNK_TOKEN]) 
                           for word in question.split()]
        question_tensor = torch.tensor([question_indices]).to(self.device)
        
        with torch.no_grad():
            # 编码器处理
            embedded = self.model.embedding(question_tensor)
            _, (hidden, cell) = self.model.encoder(embedded)
            
            # 解码器处理
            decoder_input = torch.tensor([[self.vocab.word2idx[SOS_TOKEN]]]).to(self.device)
            response = []
            used_phrases = set()  # 用于追踪已使用的短语，避免重复
            
            for _ in range(max_length):
                embedded_decoder = self.model.embedding(decoder_input)
                embedded_decoder = embedded_decoder.view(1, 1, -1)
                
                decoder_output, (hidden, cell) = self.model.decoder(
                    embedded_decoder,
                    (hidden, cell)
                )
                
                # 生成预测并应用temperature sampling
                prediction = self.model.out(decoder_output.squeeze(1))
                prediction = F.softmax(prediction / 0.7, dim=1)  # 添加temperature参数
                predicted_idx = torch.multinomial(prediction, 1).item()
                
                # 如果预测到EOS标记或生成了过多重复内容，结束生成
                if (predicted_idx == self.vocab.word2idx[EOS_TOKEN] or
                    len(response) >= 3 and ' '.join(response[-3:]) in used_phrases):
                    break
                    
                predicted_word = self.vocab.idx2word[predicted_idx]
                
                # 记录三个词的短语以检测重复
                if len(response) >= 2:
                    used_phrases.add(' '.join(response[-2:] + [predicted_word]))
                    
                response.append(predicted_word)
                decoder_input = torch.tensor([[predicted_idx]]).to(self.device)
                
            return ' '.join(response)

def prepare_data():
    """准备训练数据"""
    # 扩充对话数据集
    conversations = [
        ("你好", "你好啊，很高兴见到你"),
        ("今天天气怎么样", "今天天气很好，阳光明媚"),
        ("你叫什么名字", "我是AI助手，很高兴为你服务"),
        ("你会做什么", "我可以和你聊天，回答问题，提供帮助"),
        ("现在几点了", "抱歉，我不能获取实时时间"),
        ("你喜欢什么", "我喜欢和人类交谈，学习新知识"),
        ("再见", "再见，期待下次和你聊天"),
        ("吃饭了吗", "我是AI助手，不需要吃饭"),
        ("你是谁", "我是一个AI聊天助手，目的是帮助用户"),
        ("你很聪明", "谢谢夸奖，我会继续努力学习"),
        ("你有感情吗", "我是AI程序，不具备真实的感情"),
        ("你累吗", "我是程序，不会感到疲劳"),
        ("你会思考吗", "我可以处理信息，但不具备人类的思维能力"),
        ("你住在哪里", "我是运行在计算机中的程序"),
        ("你多大了", "我是一个新开发的AI助手"),
    ]
    
    # 构建词汇表
    vocab = Vocabulary()
    for q, a in conversations:
        for word in q.split() + a.split():
            vocab.add_word(word)
            
    # 创建数据集
    questions, answers = zip(*conversations)
    dataset = ChatDataset(questions, answers, vocab)
    
    return dataset, vocab

def train_chat_model():
    """训练聊天模型"""
    # 准备数据
    dataset, vocab = prepare_data()
    train_loader = DataLoader(
        dataset, 
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChatModel(len(vocab), embedding_dim=256, hidden_size=512)  # 增加模型容量
    
    # 训练模型
    trainer = Trainer(model, vocab, device)
    trainer.train(train_loader, num_epochs=500)  # 增加训练轮数
    
    return trainer

def chat():
    """聊天函数"""
    trainer = train_chat_model()
    
    print("开始聊天！(输入 'quit' 结束)")
    while True:
        user_input = input("你: ")
        if user_input.lower() == 'quit':
            break
            
        response = trainer.generate_response(user_input)
        print("AI: ", response)

# 添加 collate_fn 函数处理变长序列
def collate_fn(batch):
    """处理变长序列的批处理函数"""
    # 分离问题和回答
    questions = [item[0] for item in batch]
    answers = [item[1] for item in batch]
    
    # 填充序列
    questions_padded = pad_sequence(questions, batch_first=True, padding_value=0)
    answers_padded = pad_sequence(answers, batch_first=True, padding_value=0)
    
    return questions_padded, answers_padded

if __name__ == "__main__":
    chat() 