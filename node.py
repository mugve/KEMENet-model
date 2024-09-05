import scipy.io as scio  # 读取.mat文件
import time

import matplotlib.pyplot as plt
# 读取sla数据SLA_dataset.mat


SLA = scio.loadmat(r'data1.mat')
topo = scio.loadmat(r'topo2.mat')

topo = topo['topo2']
# print(topo.shape)
'''取出  lat lon  sla_1  sla_2 time0 time0  '''
sla_lat = SLA['lat']  # 纬度
print(sla_lat.shape)  #
sla_lon = SLA['lon']  # 经度
print(sla_lon.shape)  #
sla_1 = SLA['SLA_1']  # 读取的sla原始数据
print(sla_1.shape)  #

# 数据预处理-------------------------------------------------------------------------------------------------------------
import numpy as np

sla_2 = sla_1[:, :, :]  # 取days
sla_2[topo >= -200] = np.nan
print(sla_2.shape)
data = np.swapaxes(sla_2, 0, 2)
import torch

torch.manual_seed(3407)
# torch.cuda.manual_seed_all(3407)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
data = torch.from_numpy(data).float()
torch.save(data, 'data2.pt')
# 创建一个蒙版
nan_mask = torch.isnan(data).float()

nan_mask = torch.isnan(data)

data[nan_mask] = torch.nanmean(data)
import torch.nn.functional as F
a = torch.nanmean(data)

if torch.cuda.is_available():
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")
from model import *
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
torch.manual_seed(3407)

# Dataset-----------------------------------------------------------------
class MyDataset(Dataset):
    def __init__(self, data, input_sequence_length, target_sequence_length):
        self.data = data
        self.input_sequence_length = input_sequence_length
        self.target_sequence_length = target_sequence_length

    def __len__(self):
        return len(self.data) - self.input_sequence_length - self.target_sequence_length + 1

    def __getitem__(self, index):
        input_seq = self.data[index: index + self.input_sequence_length]
        target_seq = self.data[
                     index + self.input_sequence_length: index + self.input_sequence_length + self.target_sequence_length]
        return input_seq, target_seq


# data1 = padded_data
data1 = data
torch.save(data1, 'data1.pt')
# --------------------------------------------------------------------------------
batch_size = 128  # 批大小batch size
num_days = data1.size(0)  # 天数是数据的第一维
input_sequence_length = 15  # 输入天数是15也就是15天预测
target_sequence_length = 30 # 每15天预测15天
# 划分训练集和测试集------------------------------------------------------------------

split_index = 5114
print(split_index)
train_data = data1[:split_index]

val_index = 1096
print(val_index)
val_data = data1[split_index:split_index + val_index]
print(val_data.shape)
test_data = data1[split_index + val_index:]

print(test_data.shape)
# torch.save(test_data, 'test_data.pt')
# 创建 DataLoader for training data-----------------------------------------------
train_dataset = MyDataset(train_data, input_sequence_length, target_sequence_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_dataset = MyDataset(val_data, input_sequence_length, target_sequence_length)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# 定义模型----------------------------------------------------------------------------
model = R2U_Net(in_channels=15, out_channels=30).to(device)  # Adjust in_channels and out_channels accordingly


train_losses = []
val_losses = []
# # 定义 loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.1, verbose=True)
# 训练-----------------------------------  ------------------------------------------
num_epochs = 100
best_val_loss = float('inf')  # 初始化为正无穷
start_time = time.time()
for epoch in range(num_epochs):
    print("-----第{}轮训练开始------".format(epoch + 1))
    total_train_step = 1
    all_loss = 0
    for inputs, targets in train_dataloader:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        all_loss += loss
        total_train_step = total_train_step + 1
    avg_train_loss = all_loss / total_train_step
    train_losses.append(avg_train_loss.item())
    print("训练次数：{}, LOSS:{}".format(total_train_step, avg_train_loss))

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for inputs_val, targets_val in val_dataloader:
            inputs_val = inputs_val.to(device)
            targets_val = targets_val.to(device)
            outputs_val = model(inputs_val)
            val_loss = criterion(outputs_val, targets_val)
            total_val_loss += val_loss.item()

    average_val_loss = total_val_loss / len(val_dataloader)
    val_losses.append(average_val_loss)
    print("验证集 LOSS: {}".format(average_val_loss))
    # scheduler.step(average_val_loss)
    # 保存最佳模型
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        early_stop_counter = 0  # 重置早停计数器
    else:
        early_stop_counter += 1

        早停条件
     if early_stop_counter >= 5:
         print("在{}个epoch内验证集损失未减小，提前停止训练。".format(early_stop_counter))
         break
end_time = time.time()
elapsed_time = end_time - start_time
print(f"训练时间: {elapsed_time} 秒")


torch.cuda.empty_cache()
# 使用经过训练的模型对测试数据进行预测
# 为测试数据创建输入和目标序列
# 创建测试数据的 DataLoader------------------------------------------------------------------
test_dataset = MyDataset(test_data, input_sequence_length, target_sequence_length)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# 测试--------------------------------------------------------------------------------------
best_model = R2U_Net(in_channels=15, out_channels=30).to(device)
best_model.load_state_dict(torch.load('best_model.pt'))
best_model.eval()  # 设置模型为评估模式

# 使用测试集进行预测
test_predictions = []
total_test_loss = 0
start_time1 = time.time()
with torch.no_grad():
    for inputs_test, targets_test in test_dataloader:
        inputs_test = inputs_test.to(device)
        targets_test = targets_test.to(device)
        outputs_test = best_model(inputs_test)
        test_loss = criterion(outputs_test, targets_test)
        total_test_loss += test_loss.item()
        test_predictions.append(outputs_test)
end_time1 = time.time()
elapsed_time = end_time1 - start_time1
print(f"测试时间: {elapsed_time} 秒")
# 计算测试集上的平均损失
average_test_loss = total_test_loss / len(test_dataloader)
print("测试集 LOSS: {}".format(average_test_loss))

# 将预测结果转换为 PyTorch 张量
stacked_test_tensor = torch.cat(test_predictions, dim=0)
torch.save(stacked_test_tensor, 'stacked_test_tensor.pt')
print(stacked_test_tensor.shape)
