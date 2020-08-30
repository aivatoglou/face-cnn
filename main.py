import os
import cv2
import torch
import tarfile
import numpy as np
from sklearn.utils import shuffle
from model_train import train, test
from sklearn.model_selection import train_test_split
from model import Net
from dataset_object import Dataset

# dataset
data = []
age = []
tar = tarfile.open('crop_part1.tar.gz')
for member in tar.getmembers():
    f = tar.extractfile(member)
    if not f:
        continue
    age.append(int(str(member).split('_', 2)[1].replace('part1/', '')))
    content = f.read()
    face = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_GRAYSCALE)
    face = np.float32(face/255.0)
    face = cv2.resize(face, (64, 64))  
    data.append(face)

classes = []
for i in age:
    if i <= 14:
        classes.append(0)
    if (i>14) and (i<=25):
        classes.append(1)
    if (i>25) and (i<40):
        classes.append(2)
    if (i>=40) and (i<60):
        classes.append(3)
    if i>=60:
        classes.append(4)

X = np.squeeze(data)
Y = np.asarray(classes)

# Parameters
n_classes = len(np.unique(classes))
batch_size = 512
epochs = 100
lr = 0.0001

# Random shuffle data
X, Y = shuffle(X, Y)

# Train-Test-Validation split
train_valid_data = np.array((X[:-1956]))
train_valid_labels = np.array((Y[:-1956]))

test_data = np.array((X[-1956:]))
test_labels = np.array((Y[-1956:]))

train_data, valid_data, train_labels, valid_labels = train_test_split(train_valid_data, train_valid_labels, test_size=0.2)

trainSignData = Dataset(train_data, train_labels)
trainDataLoader = torch.utils.data.DataLoader(trainSignData, shuffle=True, batch_size=batch_size)

testSignData = Dataset(test_data, test_labels)
testDataLoader = torch.utils.data.DataLoader(testSignData, shuffle=True, batch_size=batch_size)

validSignData = Dataset(valid_data, valid_labels)
validDataLoader = torch.utils.data.DataLoader(validSignData, shuffle=True, batch_size=batch_size)

net = Net()
if torch.cuda.is_available():
    print('CUDA is available!  Training on GPU ...\n')
    net.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

for epoch in range(epochs):
    train(epoch, net, trainDataLoader, optimizer, criterion, validDataLoader)

test(net, testDataLoader)
