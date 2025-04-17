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
        # 上采样
        self.unsampling = nn.Conv1d(input_dim, 32, 1)

        # Transformer编码器
        self.hidden_dim = hidden_dim
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(32, num_heads, hidden_dim, dropout=dropout_rate, batch_first=True),
            num_layers
        )

        # BiLSTM参数
        self.num_layers = len(hidden_layer_sizes)
        self.bilstm_layers = nn.ModuleList()
        # 第一层BiLSTM
        self.bilstm_layers.append(nn.LSTM(32, hidden_layer_sizes[0], batch_first=True, bidirectional=True))
        # 后续BiLSTM层
        for i in range(1, self.num_layers):
            self.bilstm_layers.append(
                nn.LSTM(hidden_layer_sizes[i - 1] * 2, hidden_layer_sizes[i], batch_first=True, bidirectional=True))
        # 线性层
        self.linear = nn.Linear(hidden_layer_sizes[-1] * 2, output_dim)

    def forward(self, input_seq):
        batch_size = input_seq.size(0)
        # 上采样
        unsampling = self.unsampling(input_seq.permute(0, 2, 1))
        # Transformer 处理
        transformer_output = self.transformer(unsampling.permute(0, 2, 1))
        # 送入 BiLSTM 层
        bilstm_out = transformer_output
        for bilstm in self.bilstm_layers:
            bilstm_out, _ = bilstm(bilstm_out)
        predict = self.linear(bilstm_out[:, -1, :])
        return predict