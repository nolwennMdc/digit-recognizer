#!/usr/bin/env python
# coding: utf-8

# Digit Recognizer
# 
# Neural Network Method (https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) 
# 0.9890 precision

import codecs
import csv
import numpy as np
import pandas as pd
from PIL import Image
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (1.0,))])

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

print("create Kaggle Dataset")
class KaggleDataset(data.Dataset):
    """`Kaggle <https://www.kaggle.com/c/digit-recognizer/data>`_ Dataset.

    Args:
        train (bool, optional): If True, creates dataset from ``train.csv``,
            otherwise from ``test.csv``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    
    def __init__(self, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        
        if self.train:  
            kaggleset = pd.read_csv('data/train.csv').values          
            only_train_data = np.delete(kaggleset,0,axis=1)
            half_data = np.delete(only_train_data,np.s_[21000:42000],axis=0)
            train_reshape = half_data.reshape(21000,28,28) 
            self.train_data  = (torch.from_numpy(train_reshape)).type(torch.uint8)
            only_train_labels = np.delete(kaggleset[:,0],np.s_[21000:42000],axis=0) 
            self.train_labels = torch.from_numpy(only_train_labels)
        else:
            testset = pd.read_csv('data/test.csv').values
            test_reshape = testset.reshape(28000,28,28) 
            self.test_data = (torch.from_numpy(test_reshape)).type(torch.uint8)
            self.test_labels = torch.zeros([28000, 1], dtype=torch.uint8)
            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

print("load training dataset")
trainset = KaggleDataset(train=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=2,
                                          shuffle=True, num_workers=2)


classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9')


def imshow(img,num):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    name = 'plot/imshow'+num
    plt.savefig(name)

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images),'1')
print(' '.join('%5s' % classes[labels[j]] for j in range(2)))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,20,5) #2D convolution with a filter 5*5 on 1 input chanel, giving 20 output channels
        self.conv2 = nn.Conv2d(20, 50, 5) 
        self.fc1 = nn.Linear(50 * 4 * 4, 500) #linear transformation with 50*4*4 inputs and giving 500 outputs
        self.fc2 = nn.Linear(500, 10) 

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

startchrono = time.time()

print("Start training")
# training
for epoch in range(10):  # loop over the dataset multiple times 
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0): 
		# get the inputs       
		inputs, labels = data        
		# zero the parameter gradients       
		optimizer.zero_grad()
		# forward + backward + optimize      
		outputs = net(inputs)        
		loss = criterion(outputs, labels)       
		loss.backward()        
		optimizer.step()       
		# print statistics\n        
		running_loss += loss.item()        
		if (i+1) % 100 == 0 or (i+1) == len(trainloader): 
			# print every 100 mini-batches         
			"""print(\'[%d, %6d] loss: %.3f\' %\n                  
			(epoch + 1, i + 1, running_loss / 100))"""            
			running_loss = 0.0

endchrono = (time.time()- startchrono)/60
print('Finished Training, took (in min) : ',endchrono)

print("Load test set")
testset = KaggleDataset(train=False,transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=2,
                                         shuffle=False, num_workers=2)

print("Create submission file")
with torch.no_grad():
    with open('data/submission.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['ImageId', 'Label'])    
        counter = 0
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            for i in range(2):
                counter += 1
                label = labels[i]
                filewriter.writerow([counter, predicted[i].item()])    

print("Finished")