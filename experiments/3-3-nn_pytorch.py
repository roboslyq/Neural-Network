# 导入必要的PyTorch库
import torch                    # PyTorch基础库
import torch.nn as nn           # 神经网络模块
import torch.nn.functional as F # 函数式接口
import torch.optim as optim     # 优化器
"""
使用pytorch包重写3-2-nn_from_scratch.py逻辑
"""

# 定义目标函数：判断点(x,y)是否在单位圆内
# 返回值：如果点在圆内返回1.0，否则返回0.0
def o(x, y):
    return 1.0 if x * x + y * y < 1 else 0.0


# 生成训练样本
sample_density = 10  # 采样密度，决定了采样点的数量
# 生成输入数据：在[-2,2]x[-2,2]的正方形区域内均匀采样
xs = torch.FloatTensor([
    [-2.0 + 4 * x / sample_density, -2.0 + 4 * y / sample_density]
    for x in range(sample_density + 1)
    for y in range(sample_density + 1)
])
# 生成对应的标签：对每个采样点，计算其是否在单位圆内
ys = torch.FloatTensor([
    [o(x, y)]
    for x, y in xs 
])


# 定义神经网络模型
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        
        # 第一层全连接层：2个输入特征，4个输出特征
        self.fc1 = nn.Linear(in_features=2, out_features=4)
        # 第二层全连接层：4个输入特征，1个输出特征
        self.fc2 = nn.Linear(in_features=4, out_features=1)

    def forward(self, x):
        # 前向传播过程
        o2 = self.fc1(x)           # 第一层线性变换
        o3 = torch.sigmoid(o2)     # 第一个sigmoid激活函数
        o5 = self.fc2(o3)          # 第二层线性变换
        o4 = torch.sigmoid(o5)     # 第二个sigmoid激活函数
        return o4


# 设置随机种子，确保结果可重现
torch.manual_seed(0)

# 实例化神经网络
net = MyNet()


# 定义单步训练函数
def one_step(optimizer):
    # 清空梯度
    optimizer.zero_grad()

    # 前向传播
    output = net(xs)
    # 计算均方误差损失
    loss = F.mse_loss(output, ys)

    # 反向传播
    loss.backward()
    # 更新参数
    optimizer.step()

    return loss


# 定义训练函数
def train(epoch, learning_rate):
    """
    训练神经网络
    Args:
        epoch: 训练轮数
        learning_rate: 学习率
    """
    # 使用随机梯度下降优化器
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    # 开始训练循环
    for i in range(epoch):
        loss = one_step(optimizer)
        # 每100轮打印一次损失值
        if i == 0 or (i+1) % 100 == 0:
            print(f"第{i+1}轮 损失值：{loss:.4f}")


# 定义推理函数
def inference(x, y):
    """
    使用训练好的模型进行预测
    Args:
        x: x坐标
        y: y坐标
    Returns:
        预测该点是否在单位圆内的概率
    """
    # 设置为评估模式
    net.eval()
    # 关闭梯度计算
    with torch.no_grad():
        return net(torch.FloatTensor([[x, y]]))


# 开始训练，设置训练轮数为2000，学习率为10
train(2000, learning_rate=10)
# 测试模型：预测点(1,2)是否在单位圆内
print(inference(1, 2))
