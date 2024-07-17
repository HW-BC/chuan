# %% md
# ## 定义所有神经网络
# ### RNN
# %%
import torch
from torch import nn

class RNNModel(nn.Module):
    # 定义RNN模型
    # hidden_dim：隐藏层维度
    # num_layers：RNN层数
    # input_dim：输入维度
    # output_dim：输出维度
    def __init__(self, hidden_dim=64, num_layers=2, input_dim=1, output_dim=1):
        super(RNNModel, self).__init__()
        # batch_first=True 表示输入数据的形状为 [batch_size, sequence_length, input_dim]
        # batch_size: 批处理大小，即一次处理的数据数量,batch_size=3
        # sequence_length: 序列长度，即每个输入样本的长度,（window大小，前四天CO2的排放量）,squence_length=4
        # input_dim: 特征数量（CO2的排放量作为输入特征预测明天CO2排放量),input_dim=1
        # shape:
        # 二维：
        # [
        #     [1,2,5],
        #     [2,6,6],
        #     [3,6,9],
        #     [4,9,6],
        # ]
        # 三维：
        # [
        #     [[8],[9],[2],[3]],
        #     [[2],[8],[3],[4]],
        #     [[3],[3],[6],[9]],
        #     [[3],[3],[6],[9]],
        # ]
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# %% md
# ### GRU 结合 Transformer
# %%
class GRUModel(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=2, input_dim=1, output_dim=1):
        super(GRUModel, self).__init__()
        # GRU层
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # 创建一个多头注意力层，输入的特征维度为 hidden_dim，使用2个注意力头
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=2)
        # 层归一化层
        self.ln = nn.LayerNorm(hidden_dim)
        # 前馈神经网络，包括两层线性变换和一个ReLU激活函数
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_dim)
        )
        # 线性层，用于将隐藏状态映射到输出维度
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 初始化 GRU 隐藏状态h0,大小为 (num_layers, batch_size, hidden_dim)，并将其移动到输入 x 的设备上（CPU或GPU）
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        # 通过GRU层：将输入 x 和初始隐藏状态 h0 传入GRU层，输出 out 是GRU的输出，第二个返回值是最后一个隐藏状态（在这里未使用）
        out, _ = self.gru(x, h0)

        # # 将输出 out 传入注意力层，得到注意力加权后的输出 at_out 和注意力权重
        # at_out, _ = self.attention(out, out, out)
        # # 将注意力加权后的输出与原始输出相加（残差连接），并进行层归一化
        # out = self.ln(out + at_out)
        # # 将归一化后的输出传入前馈神经网络
        # out = self.ffn(out)
        # # 将前馈神经网络的输出传入线性层，得到最终的输出
        # out = self.ln(out + at_out)

        # 提取最后一个时间步的输出 out[:, -1, :]，并通过全连接层得到最终输出
        out = self.fc(out[:, -1, :])
        return out


# %% md
# ### LSTM
# %%
class LSTMModel(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=2, input_dim=1, output_dim=1, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        # self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        # 防止过拟合
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
            )
        self.batchnorm = nn.BatchNorm1d(hidden_dim * 2 if bidirectional else hidden_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 初始化 LSTM 隐藏状态
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_dim).to(x.device)

        # LSTM 前向传播
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出，并送入全连接层
        # out = self.fc(out[:, -1, :])
        # 防止过拟合
        out = self.dropout(self.batchnorm(out[:, -1, :]))
        out = self.fc(out)
        return out
        # return out


# %% md
# ### FC (Linear)
# %%
class FullyConnected(nn.Module):
    def __init__(self, hidden_dim=64, input_dim=1, output_dim=1, is_linear=False, window=1):
        super(FullyConnected, self).__init__()
        self.is_linear = is_linear
        self.window = window
        # 全连接层（线性层、激活层、线性层）
        self.fc1 = nn.Linear(input_dim * window, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # 单线性层
        self.only = nn.Linear(input_dim * window, output_dim)

    def forward(self, x):
        if self.is_linear:
            # 把三维变成二维
            # 最外层为batch的大小
            # 中间层为window的大小
            # 最内层为input_dim的大小(输入特征数)（特征feature是自己设置的）
            # [
            # [[584456508.7],[1067835724],[1143043447],[1045982663]],
            # [[584456508.7],[1067835724],[1143043447],[1045982663]],
            # ...]
            x = x.view(x.size(0), -1)
            # 最外层[]的维度是batch的大小（32）
            # 里层[]的维度是input_dim * window（1*4）
            # [[584456508.7,1067835724,1143043447,1045982663],[]...,[]]
            x = self.only(x)
        else:
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
        return x