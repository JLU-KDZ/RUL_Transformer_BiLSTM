import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from joblib import dump, load
import torch.utils.data as Data
import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc("font", family='Microsoft YaHei')

# 导入 Transformer_BiLSTM 模型
from Transformer_BiLSTM import Transformer_BiLSTM

# 数据加载和预处理
def dataloader(batch_size, workers=0):
    """
    加载和预处理数据集
    """
    # 加载数据集
    train_X = load('dataset/train_X')
    train_Y = load('dataset/train_Y')
    val_X = load('dataset/val_X')
    val_Y = load('dataset/val_Y')
    test_X = load('dataset/test_X')
    test_Y = load('dataset/test_Y')

    # 数据归一化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # 确保数据是三维的
    assert len(train_X.shape) == 3, "Train set must be 3D"
    assert len(val_X.shape) == 3, "Validation set must be 3D"
    assert len(test_X.shape) == 3, "Test set must be 3D"

    # 将三维数据转换为二维数据进行归一化
    train_set_2d = train_X.reshape(-1, train_X.shape[-1])
    val_set_2d = val_X.reshape(-1, val_X.shape[-1])
    test_set_2d = test_X.reshape(-1, test_X.shape[-1])

    # 应用 StandardScaler
    train_set_2d = scaler.fit_transform(train_set_2d)
    val_set_2d = scaler.transform(val_set_2d)
    test_set_2d = scaler.transform(test_set_2d)

    # 将二维数据转换回三维
    train_X = train_set_2d.reshape(train_X.shape)
    val_X = val_set_2d.reshape(val_X.shape)
    test_X = test_set_2d.reshape(test_X.shape)

    # 创建 DataLoader
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(
            torch.tensor(train_X, dtype=torch.float32),
            torch.tensor(train_Y, dtype=torch.float32)
        ),
        batch_size=batch_size,
        num_workers=workers,
        drop_last=True
    )
    val_loader = Data.DataLoader(
        dataset=Data.TensorDataset(
            torch.tensor(val_X, dtype=torch.float32),
            torch.tensor(val_Y, dtype=torch.float32)
        ),
        batch_size=batch_size,
        num_workers=workers,
        drop_last=True
    )
    test_loader = Data.DataLoader(
        dataset=Data.TensorDataset(
            torch.tensor(test_X, dtype=torch.float32),
            torch.tensor(test_Y, dtype=torch.float32)
        ),
        batch_size=batch_size,
        num_workers=workers,
        drop_last=True
    )

    return train_loader, val_loader, test_loader

# 加载数据
batch_size = 32
train_loader, val_loader, test_loader = dataloader(batch_size)

# 参数与配置
torch.manual_seed(100)  # 设置随机种子，确保实验结果可重复
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型参数
input_dim = 1    # 输入维度
hidden_layer_sizes = [32, 64]  # BiLSTM 层结构
hidden_dim = 128  # Transformer 的隐藏维度
num_layers = 1   # Transformer 编码器层数
num_heads = 2    # 多头注意力头数
output_dim = 1   # 输出维度

# 初始化模型
model = Transformer_BiLSTM(
    input_dim=input_dim,
    hidden_layer_sizes=hidden_layer_sizes,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_heads=num_heads,
    output_dim=output_dim
)

# 定义损失函数和优化器
loss_function = nn.MSELoss()  # 使用均方误差损失
learn_rate = 0.0001  # 学习率
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, betas=(0.9, 0.999))  # Adam 优化器

# 学习率调度器
from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

# 模型训练
def model_train(epochs, model, optimizer, loss_function, train_loader, val_loader, device, scheduler=None):
    """
    模型训练函数
    """
    model = model.to(device)  # 将模型移动到指定设备

    # 初始化最低 MSE 和最佳模型
    minimum_mse = float('inf')
    best_model = model

    # 记录训练和验证损失
    train_mse = []
    val_mse = []

    # 训练开始时间
    start_time = time.time()

    for epoch in range(epochs):
        # 训练模式
        model.train()
        train_mse_loss = []

        # 训练循环
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()  # 清空梯度
            y_pred = model(seq)  # 前向传播
            loss = loss_function(y_pred, labels)  # 计算损失
            train_mse_loss.append(loss.item())
            loss.backward()  # 反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()  # 更新参数

        # 计算平均训练损失
        train_av_mseloss = np.mean(train_mse_loss)
        train_mse.append(train_av_mseloss)

        print(f'Epoch: {epoch + 1:2} Train MSE Loss: {train_av_mseloss:10.8f}')

        # 验证模式
        model.eval()
        val_mse_loss = []

        with torch.no_grad():
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)
                pred = model(data)
                val_loss = loss_function(pred, label)
                val_mse_loss.append(val_loss.item())

        # 计算平均验证损失
        val_av_mseloss = np.mean(val_mse_loss)
        val_mse.append(val_av_mseloss)

        print(f'Epoch: {epoch + 1:2} Validation MSE Loss: {val_av_mseloss:10.8f}')

        # 更新最佳模型
        if val_av_mseloss < minimum_mse:
            minimum_mse = val_av_mseloss
            best_model = model
            print('Best model saved')

        # 学习率调度
        if scheduler is not None:
            scheduler.step()

    # 保存最佳模型
    torch.save(best_model, 'model/best_model_TransformerBiLSTM.pt')
    print(f'\nTraining Duration: {time.time() - start_time:.0f} seconds')

    # 绘制损失曲线
    plt.figure(figsize=(12, 6), dpi=100)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
    plt.style.use('seaborn-v0_8-whitegrid')

    # 颜色方案
    train_color = '#1f77b4'  # 蓝色
    val_color = '#ff7f0e'    # 橙色

    plt.plot(range(epochs), train_mse,
             color=train_color,
             linewidth=1.5,
             linestyle='-',
             alpha=0.8,
             label='Train MSE Loss')
    plt.plot(range(epochs), val_mse,
             color=val_color,
             linewidth=1.5,
             linestyle='-',
             alpha=0.8,
             label='Validation MSE Loss')

    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    print(f'Minimum Validation MSE: {minimum_mse}')

if __name__ == '__main__':
    # 确保在主模块中调用 freeze_support()
    from multiprocessing import freeze_support
    freeze_support()

    # 模型训练
    epochs = 50
    model_train(epochs, model, optimizer, loss_function, train_loader, val_loader, device, scheduler)