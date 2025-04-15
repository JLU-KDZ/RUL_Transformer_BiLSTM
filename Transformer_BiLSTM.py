import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib


# 设置 Matplotlib 中文字体
matplotlib.rc("font", family='Microsoft YaHei')

# Transformer-BiLSTM 模型
# 定义 LSTMModel 模型

class Transformer_BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes, hidden_dim, num_layers, num_heads, output_dim, dropout_rate=0.01):
        """
        params:
        input_dim          : 输入数据的维度
        hidden_layer_sizes : bilstm 隐藏层的数目和维度
        hidden_dim          : 注意力维度
        num_layers          : Transformer编码器层数
        num_heads           : 多头注意力头数
        output_dim         : 输出维度
        dropout_rate        : 随机丢弃神经元的概率
        """
        super().__init__()
        # 上采样操作
        self.unsampling = nn.Conv1d(input_dim, 32, 1)

        # Transformer编码器  Transformer layers
        self.hidden_dim = hidden_dim
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(32, num_heads, hidden_dim, dropout=dropout_rate, batch_first=True),
            num_layers
        )
        # self.avgpool = nn.AdaptiveAvgPool1d(9)

        # BiLSTM参数
        self.num_layers = len(hidden_layer_sizes)  # bilstm层数
        self.bilstm_layers = nn.ModuleList()  # 用于保存BiLSTM层的列表
        # 定义第一层BiLSTM
        self.bilstm_layers.append(nn.LSTM(32, hidden_layer_sizes[0], batch_first=True, bidirectional=True))
        # 定义后续的BiLSTM层
        for i in range(1, self.num_layers):
            self.bilstm_layers.append(
                nn.LSTM(hidden_layer_sizes[i - 1] * 2, hidden_layer_sizes[i], batch_first=True, bidirectional=True))

        # 定义线性层
        self.linear = nn.Linear(hidden_layer_sizes[-1] * 2, output_dim)

    def forward(self, input_seq):
        batch_size = input_seq.size(0)
        # 预处理  先进行上采样
        unsampling = self.unsampling(input_seq.permute(0, 2, 1))
        # Transformer 处理
        # 在PyTorch中，transformer模型的性能与batch_first参数的设置相关。
        # 当batch_first为True时，输入的形状应为(batch, sequence, feature)，这种设置在某些情况下可以提高推理性能。
        transformer_output = self.transformer(unsampling.permute(0, 2, 1))  # torch.Size([1, 1, 32])

        # 送入 BiLSTM 层
        # 改变输入形状，bilstm 适应网络输入[batch, seq_length, dim]
        bilstm_out = transformer_output
        for bilstm in self.bilstm_layers:
            bilstm_out, _ = bilstm(bilstm_out)  ## 进行一次BiLSTM层的前向传播  # torch.Size([1, 1, 64])
        predict = self.linear(bilstm_out[:, -1, :])  # torch.Size([1, 64]  # 仅使用最后一个时间步的输出
        return predict
