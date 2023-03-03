import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

train_dataset = datasets.MNIST(root="mnt/sda1/felix",transform = transform,train = True,download = True)
train_data = DataLoader(dataset=train_dataset,shuffle=True,batch_size=batch_size)
test_dataset = datasets.MNIST(root="mnt/sda1/felix",transform = transform,train = False,download=True)
test_data = DataLoader(dataset=test_dataset,shuffle=False,batch_size=batch_size)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=10,kernel_size=5)
        self.conv2 = torch.nn.Conv2d(in_channels=10,out_channels=20,kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320,10)

    def forward(self,x):
        x = torch.relu(self.pooling(self.conv1(x)))
        x = torch.relu(self.pooling(self.conv2(x)))
        x = x.view(-1,320)
        x = self.fc(x)
        return x

model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01,momentum=0.5)

def test():
    model.eval()
    total = 0
    y = 0
    for i,data in enumerate(test_data,0):
        inputs,labels = data
        y_pre = model(inputs)
        _,y_p = torch.max(y_pre,dim = 1)    #左边的是最大值，右边的是最大的下标
        y += (y_p == labels).sum().item()
        total += inputs.size(0)
    acc = 100 * y / total
    print("acc : %.6f",acc)

def train(epoch):
    model.train()
    loss_sum = 0
    for i,data in enumerate(train_data,0):
        inputs,labels = data
        y_pre = model(inputs)

        optimizer.zero_grad()
        loss = criterion(y_pre,labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

        if(i % 300 == 299):
            print("[%d,%5d] loss : %.3f " % (epoch,i+1,loss_sum / 300))
            loss_sum = 0
            test()

if __name__ == "__main__":
    for epoch in range(10):
        train(epoch)