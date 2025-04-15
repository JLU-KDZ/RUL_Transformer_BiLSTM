import torch
from joblib import load
import torch.utils.data as Data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from model_train import train_loader, val_loader, test_loader

# 加载模型和数据
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('model/best_model_TransformerBiLSTM.pt')
model = model.to(device)

# 预测函数
def predict(model, data_loader, device):
    true_values = []
    predicted_values = []
    with torch.no_grad():
        model.eval()
        for data, label in data_loader:
            true_values += label.tolist()
            data, label = data.to(device), label.to(device)
            pred = model(data)
            predicted_values += pred.tolist()
    return true_values, predicted_values

# 反归一化处理
def inverse_transform(scaler, data):
    return scaler.inverse_transform(data)

# 可视化结果（训练集分为两个子图）
def plot_train_results(true_values, predicted_values, title):
    # 计算中点索引
    mid_index = len(true_values) // 2

    # 创建两个子图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=100)

    # 设置全局字体和样式
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
    plt.style.use('seaborn-v0_8-whitegrid')

    # 颜色方案
    true_color = '#1f77b4'  # 蓝色
    predicted_color = '#ff7f0e'  # 橙色

    # 第一个子图（B0005）
    axes[0].plot(true_values[:mid_index-10], label='True Values', color=true_color, linewidth=1.5, alpha=0.8)
    axes[0].plot(predicted_values[:mid_index-10], label='Predicted Values', color=predicted_color, linewidth=1.5, linestyle='--', alpha=0.8)
    axes[0].set_title(f'{title} - B0005 Battery')
    axes[0].set_xlabel('Discharge Cycles')
    axes[0].set_ylabel('Capacity (Ah)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # 第二个子图（B0006）
    axes[1].plot(true_values[mid_index+10:], label='True Values', color=true_color, linewidth=1.5, alpha=0.8)
    axes[1].plot(predicted_values[mid_index+10:], label='Predicted Values', color=predicted_color, linewidth=1.5, linestyle='--', alpha=0.8)
    axes[1].set_title(f'{title} - B0006 Battery')
    axes[1].set_xlabel('Discharge Cycles')
    axes[1].set_ylabel('Capacity (Ah)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

# 可视化结果（单图）
def plot_results(true_values, predicted_values, title):
    plt.figure(figsize=(12, 6), dpi=100)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
    plt.style.use('seaborn-v0_8-whitegrid')

    # 颜色方案
    true_color = '#1f77b4'  # 蓝色
    predicted_color = '#ff7f0e'  # 橙色

    plt.plot(true_values, label='True Values', color=true_color, linewidth=1.5, alpha=0.8)
    plt.plot(predicted_values, label='Predicted Values', color=predicted_color, linewidth=1.5, linestyle='--', alpha=0.8)
    plt.title(title)
    plt.xlabel('Discharge Cycles')
    plt.ylabel('Capacity (Ah)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# 计算误差指标
def calculate_metrics(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    return mse, rmse, mae, r2

# 加载归一化模型
scaler = load('dataset/scaler_data')

# 预测训练集
train_true, train_pred = predict(model, train_loader, device)

# 反归一化处理
train_true = inverse_transform(scaler, np.array(train_true).reshape(-1, 1))
train_pred = inverse_transform(scaler, np.array(train_pred).reshape(-1, 1))

# 可视化训练集结果（分为两个子图）
plot_train_results(train_true, train_pred, 'Training Set Prediction Results')

# 计算训练集误差指标
train_mse, train_rmse, train_mae, train_r2 = calculate_metrics(train_true, train_pred)
print('Training Set R^2 Score:', train_r2)
print('-' * 70)
print('Training Set Mean Squared Error (MSE): ', train_mse)
print('Training Set Root Mean Squared Error (RMSE): ', train_rmse)
print('Training Set Mean Absolute Error (MAE): ', train_mae)

# 预测验证集
val_true, val_pred = predict(model, val_loader, device)

# 反归一化处理
val_true = inverse_transform(scaler, np.array(val_true).reshape(-1, 1))
val_pred = inverse_transform(scaler, np.array(val_pred).reshape(-1, 1))

# 可视化验证集结果
plot_results(val_true, val_pred, 'Validation Set Prediction Results - B0007 Battery')

# 计算验证集误差指标
val_mse, val_rmse, val_mae, val_r2 = calculate_metrics(val_true, val_pred)
print('Validation Set R^2 Score:', val_r2)
print('-' * 70)
print('Validation Set Mean Squared Error (MSE): ', val_mse)
print('Validation Set Root Mean Squared Error (RMSE): ', val_rmse)
print('Validation Set Mean Absolute Error (MAE): ', val_mae)

# 预测测试集
test_true, test_pred = predict(model, test_loader, device)

# 反归一化处理
test_true = inverse_transform(scaler, np.array(test_true).reshape(-1, 1))
test_pred = inverse_transform(scaler, np.array(test_pred).reshape(-1, 1))

# 可视化测试集结果
plot_results(test_true, test_pred, 'Test Set Prediction Results - B0018 Battery')

# 计算测试集误差指标
test_mse, test_rmse, test_mae, test_r2 = calculate_metrics(test_true, test_pred)
print('Test Set R^2 Score:', test_r2)
print('-' * 70)
print('Test Set Mean Squared Error (MSE): ', test_mse)
print('Test Set Root Mean Squared Error (RMSE): ', test_rmse)
print('Test Set Mean Absolute Error (MAE): ', test_mae)