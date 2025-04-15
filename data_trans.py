from datetime import datetime
from scipy.io import loadmat
from joblib import dump, load
import numpy as np


# 转换时间格式，将字符串转换成 datatime 格式
def convert_to_time(hmm):
    year, month, day, hour, minute, second = int(hmm[0]), int(hmm[1]), int(hmm[2]), int(hmm[3]), int(hmm[4]), int(
        hmm[5])
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)


# 加载 mat 文件
def loadMat(matfile):
    data = loadmat(matfile)
    print("Keys in the mat file:", data.keys())  # 打印 mat 文件中的键
    # 找到包含实际数据的键
    for key in data.keys():
        if isinstance(data[key], np.ndarray) and data[key].size > 0:
            col = data[key]
            break
    col = col[0][0][0][0]
    size = col.shape[0]

    data_list = []
    for i in range(size):
        k = list(col[i][3][0].dtype.fields.keys())
        d1, d2 = {}, {}
        if str(col[i][0][0]) != 'impedance':
            for j in range(len(k)):
                t = col[i][3][0][0][j][0]
                l = [t[m] for m in range(len(t))]
                d2[k[j]] = l
        # 确保提取的是标量
        type_val = np.squeeze(col[i][0])
        temp_val = np.squeeze(col[i][1])
        time_val = np.squeeze(col[i][2])

        d1['type'], d1['temp'], d1['time'], d1['data'] = str(type_val), int(temp_val), str(
            convert_to_time(time_val)), d2
        data_list.append(d1)

    return data_list


# 提取锂电池容量
def getBatteryCapacity(Battery):
    cycle, capacity = [], []
    i = 1
    for Bat in Battery:
        if Bat['type'] == 'discharge':
            capacity.append(Bat['data']['Capacity'][0])
            cycle.append(i)
            i += 1
    return [cycle, capacity]


# 从 mat 文件中读取数据，保存为python格式
filenames = ['B0005', 'B0006', 'B0007', 'B0018']  # 4 个数据集的名字

capacity, charge, discharge = {}, {}, {}
for file in filenames:
    print('Load Dataset: ' + file + '.mat')
    path = 'dataset/' + file + '.mat'
    data = loadMat(path)
    capacity[file] = getBatteryCapacity(data)  # 放电时的容量数据

# 保存数据
dump(capacity, 'dataset/capacity')
