import math
from random import seed, random
"""
不使用pytorch等工具库，从零开始实现一个神经网络
目标：实现一个简单的神经网络来判断点是否在单位圆内
代码实现的核心逻辑：
目标任务：
    实现一个神经网络来判断平面上的点是否在单位圆内
    输入：点的(x,y)坐标
    输出：0到1之间的值，表示点在圆内的概率
网络结构：
    输入层：2个神经元（x和y坐标）
    隐藏层：4个神经元
    输出层：1个神经元
    激活函数：sigmoid
核心组件：
    Neuron类：实现单个神经元的前向传播和反向传播
    MyNet类：组织多层神经元形成网络
    损失函数：使用均方误差(MSE)
训练过程：
    在[-2,2]×[-2,2]区域内均匀采样点作为训练数据
    使用随机梯度下降优化参数
    训练2000轮，每100轮打印一次损失
创新点：
    完全从零实现，不依赖深度学习框架
    实现了完整的前向传播和反向传播
    手动实现了参数更新和梯度计算
"""

def o(x, y):
    """目标函数：判断点(x,y)是否在单位圆内
    Args:
        x, y: 点的坐标
    Returns:
        1.0: 点在单位圆内
        0.0: 点在单位圆外
    """
    return 1.0 if x*x + y*y < 1 else 0.0


# 生成训练样本
sample_density = 10  # 采样密度
# 在[-2,2]x[-2,2]的正方形区域内均匀采样点
xs = [
    [-2.0 + 4 * x/sample_density, -2.0 + 4 * y/sample_density]
    for x in range(sample_density+1)
    for y in range(sample_density+1)
]
# 生成训练数据集：(x坐标, y坐标, 是否在圆内)
dataset = [
    (x, y, o(x, y))
    for x, y in xs
]


def sigmoid(x):
    """sigmoid激活函数
    Args:
        x: 输入值
    Returns:
        sigmoid(x) = 1/(1+e^(-x))
    """
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    """sigmoid函数的导数
    Args:
        x: 输入值
    Returns:
        sigmoid'(x) = sigmoid(x)(1-sigmoid(x))
    """
    _output = sigmoid(x)
    return _output * (1 - _output)


seed(0)  # 设置随机种子以确保结果可重现


class Neuron:
    """神经元类
    实现单个神经元的前向传播和反向传播
    """
    def __init__(self, num_inputs):
        """初始化神经元
        Args:
            num_inputs: 输入特征的数量
        """
        # 随机初始化权重为[-0.5,0.5]之间的值
        self.weights = [random()-0.5 for _ in range(num_inputs)]
        self.bias = 0.0  # 偏置初始化为0

        # 缓存前向传播的中间值，用于反向传播
        self.z_cache = None  # 缓存z = wx + b
        self.inputs_cache = None  # 缓存输入值

    def forward(self, inputs):
        """前向传播
        Args:
            inputs: 输入特征列表
        Returns:
            神经元的输出：sigmoid(wx + b)
        """
        assert len(inputs) == len(inputs)
        self.inputs_cache = inputs

        # 计算z = wx + b
        self.z_cache = sum([
            i * w
            for i, w in zip(inputs, self.weights)
        ]) + self.bias
        # 返回sigmoid(z)
        return sigmoid(self.z_cache)

    def zero_grad(self):
        """清空梯度"""
        self.d_weights = [0.0 for w in self.weights]
        self.d_bias = 0.0

    def backward(self, d_a):
        """反向传播，计算梯度
        Args:
            d_a: 输出对应的梯度
        Returns:
            输入对应的梯度列表
        """
        # 计算sigmoid的梯度
        d_loss_z = d_a * sigmoid_derivative(self.z_cache)
        # 累加偏置的梯度
        self.d_bias += d_loss_z
        # 计算权重的梯度
        for i in range(len(self.inputs_cache)):
            self.d_weights[i] += d_loss_z * self.inputs_cache[i]
        # 返回输入的梯度
        return [d_loss_z * w for w in self.weights]

    def update_params(self, learning_rate):
        """更新参数
        Args:
            learning_rate: 学习率
        """
        # 使用梯度下降更新参数
        self.bias -= learning_rate * self.d_bias
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.d_weights[i]

    def params(self):
        """返回神经元的所有参数"""
        return self.weights + [self.bias]


class MyNet:
    """神经网络类
    实现一个具有一个隐藏层的神经网络
    """
    def __init__(self, num_inputs, hidden_shapes):
        """初始化神经网络
        Args:
            num_inputs: 输入特征数量
            hidden_shapes: 隐藏层神经元数量列表
        """
        layer_shapes = hidden_shapes + [1]  # 添加输出层（1个神经元）
        input_shapes = [num_inputs] + hidden_shapes  # 每层的输入维度
        # 创建每一层的神经元
        self.layers = [
            [
                Neuron(pre_layer_size)
                for _ in range(layer_size)
            ]
            for layer_size, pre_layer_size in zip(layer_shapes, input_shapes)
        ]

    def forward(self, inputs):
        """前向传播
        Args:
            inputs: 输入特征
        Returns:
            网络的输出
        """
        for layer in self.layers:
            inputs = [
                neuron.forward(inputs)
                for neuron in layer
            ]
        return inputs[0]  # 返回最后一层的输出

    def zero_grad(self):
        """清空所有参数的梯度"""
        for layer in self.layers:
            for neuron in layer:
                neuron.zero_grad()

    def backward(self, d_loss):
        """反向传播
        Args:
            d_loss: 损失函数的梯度
        """
        d_as = [d_loss]  # 初始梯度
        # 从后向前传播梯度
        for layer in reversed(self.layers):
            da_list = [
                neuron.backward(d_a)
                for neuron, d_a in zip(layer, d_as)
            ]
            # 合并多个神经元的梯度
            d_as = [sum(da) for da in zip(*da_list)]

    def update_params(self, learning_rate):
        """更新所有参数
        Args:
            learning_rate: 学习率
        """
        for layer in self.layers:
            for neuron in layer:
                neuron.update_params(learning_rate)

    def params(self):
        """返回网络的所有参数"""
        return [[neuron.params() for neuron in layer]
                for layer in self.layers]


def square_loss(predict, target):
    """均方损失函数"""
    return (predict-target)**2


def square_loss_derivative(predict, target):
    """均方损失函数的导数"""
    return 2 * (predict-target)


# 创建神经网络：2个输入，4个隐藏神经元，1个输出
net = MyNet(2, [4])
print(net.forward([0, 0]))  # 测试前向传播
targets = [z for x, y, z in dataset]  # 提取目标值


def one_step(learning_rate):
    """执行一步训练
    Args:
        learning_rate: 学习率
    Returns:
        平均损失
    """
    net.zero_grad()  # 清空梯度

    loss = 0.0
    num_samples = len(dataset)
    # 对每个样本进行训练
    for x, y, z in dataset:
        predict = net.forward([x, y])  # 前向传播
        loss += square_loss(predict, z)  # 计算损失

        # 反向传播（除以样本数以计算平均梯度）
        net.backward(square_loss_derivative(predict, z) / num_samples)

    net.update_params(learning_rate)  # 更新参数
    return loss / num_samples  # 返回平均损失


def train(epoch, learning_rate):
    """训练函数
    Args:
        epoch: 训练轮数
        learning_rate: 学习率
    """
    for i in range(epoch):
        loss = one_step(learning_rate)
        if i == 0 or (i + 1) % 100 == 0:
            print(f"{i + 1} {loss:.4f}")


def inference(x, y):
    """推理函数
    Args:
        x, y: 输入坐标
    Returns:
        预测该点是否在圆内的概率
    """
    return net.forward([x, y])


# 开始训练
train(2000, learning_rate=10)
inference(1, 2)  # 测试推理
