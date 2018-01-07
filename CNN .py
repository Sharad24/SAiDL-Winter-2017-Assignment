
# coding: utf-8

# In[ ]:

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import torch
get_ipython().magic('matplotlib inline')


# In[2]:

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optimization
import torchvision.transforms as transforms
import torchvision


# In[3]:

from PIL import Image


# In[4]:

from imutils import paths
import os
import random
def concatinate(list):
    result = ''
    for element in list:
        result += element
        if element != '':
            if element != list[-1]:
                result += ' '
            else:
                result += ''
    return result


# In[5]:

classes = ['cat', 'dog']    
breeds = ['Siamese', 'leonberger', 'shiba inu', 'pomeranian', 'Bengal', 'Bombay', 'British Shorthair',
           'Sphynx', 'chihuahua', 'Persian', 'scottish terrier', 'german shorthaired', 'keeshond',
           'Abyssinian', 'saint bernard', 'Ragdoll', 'Maine Coon', 'american pit bull terrier', 
           'wheaten terrier', 'samoyed', 'great pyrenees', 'pug', 'newfoundland', 'yorkshire terrier', 
           'staffordshire bull terrier', 'english cocker spaniel', 'japanese chin', 'havanese', 
           'miniature pinscher', 'beagle', 'basset hound', 'boxer', 'Russian Blue', 'Birman', 'Egyptian Mau',
           'american bulldog', 'english setter']
batch_size = 1


# In[6]:

transformations = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])


# In[7]:

class Convolutional_Neural_Network(nn.Module):
    def __init__(self):
        super(Convolutional_Neural_Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 8, stride = 6, padding = 4)
        self.conv2 = nn.Conv2d(64, 40, 5, stride = 4, padding = 2)
        self.conv3 = nn.Conv2d(40, 20, 3, stride = 2)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(20, 2000)
        self.fc2 = nn.Linear(2000,300)
        self.fc3 = nn.Linear(300, 37)
        
    def forward(self,x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 20)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
Instance = Convolutional_Neural_Network()


# In[8]:

Loss = nn.CrossEntropyLoss()
optimizer = optimization.SGD(Instance.parameters(), lr = 0.001, momentum = 0.5)
data11 = pd.read_csv('annotations/trainval.csv', delim_whitespace=True)['Name']


# In[10]:

data11 = data11.sample(frac=1)
total_epochs = 50
batch_size = 1
epoch_loss_data = []
for epoch in range(total_epochs):
    running_loss = 0
    for u ,element in enumerate(data11):
        image_name = element
        image = cv2.imread('images/' + image_name + '.jpg', cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (500,500))
        image = np.reshape(image, (500,500,3))
        image = Image.fromarray(image)
        image = transformations(image)
        image = np.array(image)
        image = np.reshape(image, (batch_size,3,500,500))
        image = torch.Tensor(image)
        inputs = image
        image_name = image_name.split('_')[:-1]
        image_name = concatinate(image_name)
        labels = []
        for w, element in enumerate(breeds):
            if element == image_name:
                labels.append(w)
        labels = torch.Tensor(labels).type(torch.LongTensor)
        
        labels = labels.view(batch_size)
        
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = Instance(inputs)
        loss = Loss(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
     
    epoch_loss_data.append(running_loss)
    print(epoch, running_loss)


# In[11]:

data2 = pd.read_csv('annotations/test.csv', delim_whitespace = True)['Name']


# In[14]:

plt.plot(np.arange(1,51), epoch_loss_data)


# In[36]:

correct = 0
total = 0
data2 = pd.read_csv('annotations/test.csv', delim_whitespace=True)['Name']
for u ,element in enumerate(data2):
    image_name = element
    image = cv2.imread('images/' + image_name + '.jpg', cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (500,500))
    image = np.reshape(image, (500,500,3))
    image = Image.fromarray(image)
    image = transformations(image)
    image = np.array(image)
    image = np.reshape(image, (batch_size,3,500,500))
    image = torch.Tensor(image)
    inputs = image
    image_name = image_name.split('_')[:-1]
    image_name = concatinate(image_name)
    labels = []
    for w, element in enumerate(breeds):
        if element == image_name:
            labels.append(w)
    labels = torch.Tensor(labels).type(torch.LongTensor)
        
    labels = labels.view(batch_size)
        
    inputs, labels = Variable(inputs), Variable(labels)
    outputs = Instance(inputs)
    _, predicted = torch.max(outputs, 1)
    total += 1
    correct = 0
    if predicted.data[0] == labels.data[0]:
        correct+=1
    else:
        correct+=0

        
    
    if u%200 == 199:
        print(u, 'done')


# In[43]:

print('The accuracy over the test dataset is', 100*(correct/total))


# In[ ]:



