import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from joblib import dump, load
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 读取数据
capacity_path = os.path.join('dataset', 'capacity')  # 假设 capacity 是 .joblib 文件
try:
    capacity = load(capacity_path)
except FileNotFoundError:
    print(f"Error: File {capacity_path} does not exist")
    exit()

Battery_list = ['B0005', 'B0006', 'B0007', 'B0018']  # 4 个数据集的名字

# 绘制容量退化曲线
plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
plt.style.use('seaborn-v0_8-whitegrid')

# 自定义颜色和标记
colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']
marker_list = ['x', '+', '.', '*']

for idx, name in enumerate(Battery_list):
    data = capacity[name]
    plt.plot(data[0], data[1],
             marker=marker_list[idx],
             color=colors[idx],
             markersize=8,
             linestyle='-',
             linewidth=1.5,
             alpha=0.7,
             label=name)

plt.xlabel('Discharge Cycles')
plt.ylabel('Capacity (Ah)')
plt.title('Capacity Degradation at Ambient Temperature of 24°C')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 数据预处理
def make_data_labels(x_data, y_label):
    '''
        返回 x_data: 数据集     torch.tensor
            y_label: 对应标签值  torch.tensor
    '''
    x_data = torch.tensor(x_data).float()
    y_label = torch.tensor(y_label).float()
    return x_data, y_label

def data_window_maker(time_series, window_size):
    '''
        参数:
        time_series: 时间序列数据(为numpy数组格式)
        window_size: 滑动窗口大小

        返回:
        data_x: 特征数据
        data_y: 标签数据
    '''
    data_x = []
    data_y = []
    data_len = time_series.shape[0]
    for i in range(data_len - window_size):
        data_x.append(time_series[i:i + window_size, :])  # 输入特征
        data_y.append(time_series[i + window_size, :])  # 输出标签
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    data_x, data_y = make_data_labels(data_x, data_y)
    return data_x, data_y

def make_wind_dataset(data, window_size):
    '''
        参数:
        data: 数据集(为numpy数组格式)
        window_size: 滑动窗口大小

        返回:
        data_x: 特征数据
        data_y: 标签数据
    '''
    data_x, data_y = data_window_maker(data, window_size)
    return data_x, data_y

# 数据归一化和滑动窗口处理
# Training Set (B0005, B0006)
train_data_list = []
for name in ['B0005', 'B0006']:
    target_data = capacity[name][1]
    target_data = np.array(target_data).reshape(-1, 1)  # 转换为 numpy 数组
    train_data_list.append(target_data)

# Combine the two datasets
train_data = np.concatenate(train_data_list, axis=0)

# Validation Set (B0007)
val_data = capacity['B0007'][1]
val_data = np.array(val_data).reshape(-1, 1)  # 转换为 numpy 数组

# Test Set (B0018)
test_data = capacity['B0018'][1]
test_data = np.array(test_data).reshape(-1, 1)  # 转换为 numpy 数组

# Normalize the training set
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
dump(scaler, os.path.join('dataset', 'scaler_data'))  # 保存归一化模型

# Apply the same normalization to validation and test sets
val_data = scaler.transform(val_data)
test_data = scaler.transform(test_data)

# Sliding window processing
window_size = 1

# 处理训练集
train_X, train_Y = make_wind_dataset(train_data, window_size)

# 处理验证集
val_X, val_Y = make_wind_dataset(val_data, window_size)

# 处理测试集
test_X, test_Y = make_wind_dataset(test_data, window_size)

# 保存数据集
dump(train_X, os.path.join('dataset', 'train_X'))
dump(train_Y, os.path.join('dataset', 'train_Y'))
dump(val_X, os.path.join('dataset', 'val_X'))
dump(val_Y, os.path.join('dataset', 'val_Y'))
dump(test_X, os.path.join('dataset', 'test_X'))
dump(test_Y, os.path.join('dataset', 'test_Y'))

print('Data Shapes:')
print("Training Set:", train_X.size(), train_Y.size())
print("Validation Set:", val_X.size(), val_Y.size())
print("Test Set:", test_X.size(), test_Y.size())