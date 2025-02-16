"""
手写数字生成模型 - 详细注释版
目标：使用VAE(变分自编码器)实现MNIST手写数字的生成和识别

基本概念：
- VAE: 变分自编码器，一种生成模型
- 编码器: 将输入图像压缩为低维表示
- 解码器: 将低维表示还原为图像
- 分类器: 识别图像中的数字

主要功能：
1. 训练：学习手写数字的特征
2. 生成：创建新的手写数字图像
3. 识别：对输入的手写数字图片进行识别
"""

from pathlib import Path  # 处理文件路径
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 常用函数
from torch.utils.data import DataLoader  # 数据加载器
import torchvision                  # 计算机视觉工具
from torchvision import transforms  # 图像变换工具

from tqdm import tqdm  # 进度条
import pandas as pd  # 数据处理
import matplotlib.pyplot as plt  # 绘图
from PIL import Image  # 图像处理
import numpy as np  # 数值计算


# 数据预处理：将图像转换为张量
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为PyTorch张量，并归一化到[0,1]
])

# 加载MNIST数据集
# MNIST是一个手写数字数据集，包含0-9的手写数字图片
train_dataset = torchvision.datasets.MNIST(
    root='datasets',            # 数据集保存路径
    train=True,  # 训练集
    transform=transform,  # 使用上面定义的转换
    download=True  # 如果没有则下载
)
test_dataset = torchvision.datasets.MNIST(
    root='datasets',  # 数据集保存路径
    train=False,  # 测试集
    transform=transform,  # 使用上面定义的转换
    download=True  # 如果没有则下载
)


class Datasets:
    """数据集管理类：处理数据的加载和批处理"""
    def __init__(self, batch_size):
        # 创建训练数据加载器
        self.train_loader = DataLoader(
            train_dataset,  # 训练数据集
            batch_size=batch_size,  # 每批处理的样本数
            shuffle=True  # 随机打乱数据
        )
        # 创建测试数据加载器
        self.test_loader = DataLoader(
            test_dataset,  # 测试数据集
            batch_size=batch_size*2,  # 测试时可以用更大的批次
            shuffle=False  # 测试时不需要打乱
        )


class VAE(nn.Module):
    """变分自编码器模型：包含编码器、解码器和分类器"""
    def __init__(self, latent_dim=20):
        """初始化模型结构
        Args:
            latent_dim: 隐空间维度，即压缩后的特征维度
        """
        super(VAE, self).__init__()
        
        # 编码器网络：将28x28的图像压缩为低维特征
        self.encoder = nn.Sequential(
            # 第一个卷积层：1通道->32通道，图像大小变为14x14
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  # 激活函数
            # 第二个卷积层：32通道->64通道，图像大小变为7x7
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),  # 展平为一维
            nn.Linear(64 * 7 * 7, 256),  # 全连接层
            nn.ReLU()
        )
        
        # 均值和方差预测层：用于生成隐空间表示
        self.fc_mu = nn.Linear(256, latent_dim)  # 预测均值
        self.fc_var = nn.Linear(256, latent_dim)  # 预测方差
        
        # 解码器网络：将低维特征还原为28x28的图像
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),  # 重新变为3维
            # 反卷积层：逐步还原图像尺寸
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # 输出范围压缩到[0,1]
        )

        # 分类器层：用于识别数字（0-9）
        # nn.Sequential 是 PyTorch 中的一个容器模块，它可以按照顺序依次执行一系列的神经网络层。
        self.classifier = nn.Sequential(
            # 第一个全连接层，将输入维度从 latent_dim 映射到 64
            nn.Linear(latent_dim, 64),
            # 激活函数，引入非线性
            nn.ReLU(),
            # 第二个全连接层，将维度从 64 映射到 10
            nn.Linear(64, 10),  # 10个数字类别
            # 对数 softmax 函数，输出概率分布
            nn.LogSoftmax(dim=1)  # 输出概率分布
        )

    def encode(self, x):
        """编码过程：将输入图像转换为隐空间表示"""
        result = self.encoder(x)
        mu = self.fc_mu(result)  # 均值
        log_var = self.fc_var(result)  # 对数方差
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """重参数化技巧：使得反向传播可行"""
        std = torch.exp(0.5 * log_var)  # 标准差
        eps = torch.randn_like(std)  # 随机噪声
        return mu + eps * std  # 采样

    def decode(self, z):
        """解码过程：将隐空间表示转换回图像"""
        return self.decoder(z)

    def forward(self, x):
        """前向传播：完整的编码-解码过程"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def classify(self, x):
        """分类过程：识别图像中的数字"""
        mu, _ = self.encode(x)
        return self.classifier(mu)


class Trainer:
    """训练器类"""
    def __init__(self, datasets, model, optimizer, results_path="D:\\ai\\models"):
        self.datasets = datasets
        self.model = model
        self.optimizer = optimizer
        self.results_path = Path(results_path)
        self.train_df = None

        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("using device:", self.device)
        model.to(self.device)

        # 添加分类损失
        self.clf_criterion = nn.NLLLoss()

    def loss_function(self, recon_x, x, mu, log_var):
        """VAE损失函数：重建损失 + KL散度"""
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD

    def train_epoch(self, epoch, num_epochs):
        """训练一个epoch"""
        self.model.train()
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(tqdm(self.datasets.train_loader, 
                                                       desc=f"Epoch {epoch}/{num_epochs}")):
            data = data.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            
            # VAE损失
            recon_batch, mu, log_var = self.model(data)
            vae_loss = self.loss_function(recon_batch, data, mu, log_var)
            
            # 分类损失
            clf_output = self.model.classifier(mu)
            clf_loss = self.clf_criterion(clf_output, labels)
            
            # 总损失
            loss = vae_loss + clf_loss
            
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

        return train_loss / len(self.datasets.train_loader.dataset)

    def test(self):
        """测试模型"""
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in self.datasets.test_loader:
                data = data.to(self.device)
                recon_batch, mu, log_var = self.model(data)
                test_loss += self.loss_function(recon_batch, data, mu, log_var).item()

        test_loss /= len(self.datasets.test_loader.dataset)
        return test_loss

    def train(self, num_epochs):
        """完整训练过程"""
        losses = []
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch, num_epochs)
            test_loss = self.test()
            print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
            losses.append([train_loss, test_loss])

        self.save_model()
        self.train_df = pd.DataFrame(losses, columns=['train_loss', 'test_loss'])
        self.train_df.to_csv(self.results_path/ 'train.csv', index=False)

    def save_model(self):
        """保存模型"""
        self.results_path.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), self.results_path/ 'model.pth')

    def plot_results(self):
        """绘制训练结果"""
        plt.figure(figsize=(10, 4))
        plt.plot(self.train_df['train_loss'], label='Train Loss')
        plt.plot(self.train_df['test_loss'], label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()


class Generator:
    """图像生成器类"""
    def __init__(self, model_path):
        """初始化生成器"""
        self.device = torch.device("cpu")
        self.model = VAE().to(self.device)
        self.model.load_state_dict(torch.load(model_path / 'model.pth'))
        self.model.eval()

    def generate(self, num_samples=10):
        """生成手写数字图像"""
        with torch.no_grad():
            # 从标准正态分布采样
            z = torch.randn(num_samples, 20).to(self.device)
            # 通过解码器生成图像
            samples = self.model.decode(z)
            return samples

    def interpolate(self, start_idx=2, end_idx=8, steps=10):
        """在隐空间中进行插值生成"""
        with torch.no_grad():
            # 获取两个真实图像
            img1 = test_dataset[start_idx][0].unsqueeze(0).to(self.device)
            img2 = test_dataset[end_idx][0].unsqueeze(0).to(self.device)
            
            # 编码到隐空间
            mu1, _ = self.model.encode(img1)
            mu2, _ = self.model.encode(img2)
            
            # 在隐空间中进行线性插值
            vectors = []
            for alpha in torch.linspace(0, 1, steps):
                vector = mu1 * (1-alpha) + mu2 * alpha
                vectors.append(vector)
            
            # 解码生成图像
            return self.model.decode(torch.cat(vectors))


def train():
    """训练函数"""
    torch.manual_seed(42)
    
    # 创建数据集和模型
    datasets = Datasets(batch_size=128)
    model = VAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 训练模型
    trainer = Trainer(datasets, model, optimizer)
    trainer.train(num_epochs=10)  # 增加训练轮数以提高性能
    trainer.plot_results()


def generate():
    """生成示例"""
    generator = Generator(Path("results"))
    
    # 生成随机样本
    samples = generator.generate(10)
    plt.figure(figsize=(10, 1))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(samples[i].squeeze(), cmap='gray')
        plt.axis('off')
    plt.show()
    
    # 生成插值样本
    interpolations = generator.interpolate()
    plt.figure(figsize=(10, 1))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(interpolations[i].squeeze(), cmap='gray')
        plt.axis('off')
    plt.show()


# 添加测试图片预处理函数
def preprocess_image(image_path):
    """预处理本地图片
    Args:
        image_path: 图片路径
    Returns:
        处理后的张量
    """
    # 打开并转换为灰度图
    image = Image.open(image_path).convert('L')
    # 调整大小为28x28
    image = image.resize((28, 28))
    # 转换为numpy数组并归一化
    image_array = np.array(image) / 255.0
    # 转换为张量并添加维度
    image_tensor = torch.FloatTensor(image_array).unsqueeze(0).unsqueeze(0)
    return image_tensor


# 添加测试函数
def test_image(image_path):
    """测试单张图片
    Args:
        image_path: 图片路径
    """
    # 加载模型
    device = torch.device("cpu")
    model = VAE().tod(device)
    model.load_state_dict(torch.load("D:\\ai\\models\\model.pth"))
    model.eval()

    # 处理图片
    image_tensor = preprocess_image(image_path)
    
    # 进行预测
    with torch.no_grad():
        # 获取分类结果
        output = model.classify(image_tensor)
        predicted_digit = output.argmax(dim=1).item()
        print("预测结果的数字: %d"  % predicted_digit)
        confidence = torch.exp(output[0][predicted_digit]).item()

        # 重建图像
        recon, _, _ = model(image_tensor)

    # 显示结果
    plt.figure(figsize=(8, 3))
    
    # 显示原始图片
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image_tensor.squeeze(), cmap='gray')
    plt.axis('off')
    
    # 显示重建图片
    plt.subplot(1, 2, 2)
    plt.title('Reconstructed Image')
    plt.imshow(recon.squeeze().detach().numpy(), cmap='gray')
    plt.axis('off')
    
    plt.suptitle(f'Predicted Digit: {predicted_digit} (Confidence: {confidence:.2%})')
    plt.show()


if __name__ == "__main__":
    # 首先重新训练模型
    # train()
    
    # 等待训练完成后，再测试图片
    test_image("C:\\Users\\Administrator\\Desktop\\mnist_jpg\\test_0_7.jpg") 
    # test_image("C:\\Users\\Administrator\\Desktop\\mnist_jpg\\test_20_9.jpg") 