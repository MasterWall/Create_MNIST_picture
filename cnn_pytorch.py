import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as Data

EPOCH =  3
BATCH_SIZE= 50
LR= 0.001
DOWNLOAD_MNIST = False


'''
对于 DOWNLOAD_MINIST 这个变量，是函数的torchvision.datasets.MINIST()函数里面的一个参数，
如果为True表示从网上下载该数据集并放进指定目录，如果自己已经下载了该数据集，则修改为False，不需要去重新下载。
'''
train_data = torchvision.datasets.MNIST(
    root = './MINIST',                                  #数据集的位置
    train = False,                                      #如果为True则为训练集，如果为False则为测试集
    transform = torchvision.transforms.ToTensor(),      #将图片转化成取值[0,1]的Tensor用于网络处理
    download=DOWNLOAD_MNIST
)

# plot one example
print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.imshow(train_data.train_data[50].numpy(),cmap='Greys')
plt.title('%i'%train_data.train_labels[50])
plt.show()



train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
test_data = torchvision.datasets.MNIST(root='./MINIST',train=False)

#只有在训练的时候才会自动压缩，所以这里采用手动压缩
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]
# print(test_x.shape)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),                              #维度变换(1,28,28) --> (16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)     #维度变换(16,28,28) --> (16,14,14)
        )
        self.conv2 = nn.Sequential(
	        nn.Conv2d(
		        in_channels=16,
		        out_channels=32,
		        kernel_size=3,
		        stride=1,
		        padding=1
	        ),                              #维度变换(16,14,14) --> (32,14,14)
	        nn.ReLU(),
	        nn.MaxPool2d(kernel_size=2)     #维度变换(32,14,14) --> (32,7,7)
        )
        self.output = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
	    out = self.conv1(x)                 #维度变换(Batch,1,28,28) --> (Batch,16,14,14)
	    out = self.conv2(out)               #维度变换(Batch,16,14,14) --> (Batch,32,7,7)
	    out = out.view(out.size(0), -1)     #维度变换(Batch,32,14,14) --> (Batch,32*14*14)||将其展平
	    out = self.output(out)
		return out

cnn = CNN()
print(cnn)
optimizer = torch.optim.Adam(cnn.parameters(),lr=LR,)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step ,(b_x,b_y) in enumerate(train_loader):
        # b_x = x
        # b_y = y

        output = cnn(b_x)
        loss = loss_func(output,b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

		if step%50 ==0:
            test_output = cnn(test_x)
            # test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# torch.save(cnn,'cnn_minist.pkl')
print('finish training')

# print('load cnn model')
# cnn1 = torch.load('cnn_minist.pkl')
# test_output = cnn1(test_x[:50])
# pred_y = torch.max(test_output, 1)[1].data.numpy()
# print(pred_y, 'prediction number')
# print(test_y[:50].numpy(), 'real number')
