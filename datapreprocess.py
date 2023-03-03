import os

os.makedirs(os.path.join('..','data'),exist_ok = True)
data_file = os.path.join('..','data','house_tiny.csv')
with open(data_file,'w') as f:
    f.write("NumRooms,Alley,Price\n")
    f.write("NA,Pave,127500\n")
    f.write("2,NA,106000\n")
    f.write("4,NA,1781000\n")
    f.write("NA,NA,140000\n")

import pandas as pd

data = pd.read_csv(data_file)
print(data)

inputs,outputs = data.iloc[:,0:2],data.iloc[:,-1]
print(inputs)
print(outputs)

#option 1
# inputs = inputs.fillna(inputs.mean(numeric_only = True))
# print(inputs)

#option 2
numeric_cols = inputs.select_dtypes(include='number').columns
#store the name of the column
inputs[numeric_cols] = inputs[numeric_cols].fillna(inputs[numeric_cols].mean())
print(inputs)

#option 1
# inputs = pd.get_dummies(inputs,dummy_na = True)
# print(inputs)
# NumRooms Alley
# 0       3.0  Pave
# 1       2.0   NaN
# 2       4.0   NaN
# 3       3.0   NaN
#option 2
inputs = pd.get_dummies(inputs,dummy_na = False)
print(inputs)
# NumRooms  Alley_Pave
# 0       3.0           1
# 1       2.0           0
# 2       4.0           0
# 3       3.0           0

import torch

X,y = torch.tensor(inputs.values),torch.tensor(outputs.values)
print(X)
print(y)

