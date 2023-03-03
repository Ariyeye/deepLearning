import torch

randn = torch.randn(2,3,4) #randn 生成的是一个均值为0，方差为1的矩阵
print(randn)

a = torch.arange(3).reshape(3,1)
b = torch.arange(3).reshape(1,3)

print(a)
print(b)
print(a + b)