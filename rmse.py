import torch
import numpy as np
import math

da = torch.load('data2.pt').to('cpu')

denor = torch.load('stacked_test_tensor.pt', map_location=torch.device('cpu'))
input_sequence_length = 15

a = denor.shape[0]
b = denor.shape[1]
print(a)
c= b-1
sla = torch.zeros((a, b, 80, 120))


cc = torch.zeros((b, a))
rmse = torch.zeros((b, a))


def mean2(x):
    y = np.sum(x) / np.size(x)
    return y


def corr2(a, b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum())
    return r



print('-----------------------')
da = da[6210 + input_sequence_length:]
# da = da[input_sequence_length:]
sla = torch.zeros((a, b, 80, 120))
from scipy.stats import pearsonr

for i in range(b):
    if i < c:
        sla[:, i, :, :] = da[i:-c + i, :, :]
    else:
        sla[:, i, :, :] = da[i:, :, :]

cc = torch.zeros((b, a))
rmse = torch.zeros((b, a))
test = torch.zeros((a, b, 80, 120))
for i in range(b):  # 15
    for j in range(a):  # 1177
        # 创建掩码，用于处理NaN值
        mask1 = ~torch.isnan(sla[j, i])

        # 提取非NaN值的元素
        non_nan_data1 = torch.masked_select(sla[j, i], mask1)
        non_nan_data2 = torch.masked_select(denor[j, i], mask1)

        # 计算相关系数
        correlation_coefficient = corr2(non_nan_data1.numpy(), non_nan_data2.numpy())
        cc[i, j] = correlation_coefficient
        squared_diff = (non_nan_data1 - non_nan_data2) ** 2
        rm = torch.sqrt(torch.nanmean(squared_diff))
        rmse[i, j] = rm
        # 将 denor 中对应 mask1 位置的值替换为 NaN
        denor[j, i][~mask1] = float('nan')
        # 将更新后的 denor 保存到 test 中
        test[j, i] = denor[j, i]
torch.save(cc, 'cc.pt')
for i in range(b):
    print(torch.mean(cc[i, :]))
print('-----------------------')
for i in range(b):
    print(torch.mean(rmse[i, :]))
torch.save(test, 'test.pt')
torch.save(rmse,'rmse.pt')
