import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import random
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
    
learning_rate = 1e-4
training_epochs = 500
batch_size = 10

os.chdir('C:/Users/CSD/Desktop/CPR_project/cnn')

from dataset import BasicDataset

os.chdir('C:/Users/CSD/Desktop/CPR_project/cnn')

mitdb_train = BasicDataset(root_dir='./data_mV', train=True)
data_loader = torch.utils.data.DataLoader(dataset=mitdb_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
mitdb_test = BasicDataset(root_dir='./data_mV', train=False)
data_loader_test = torch.utils.data.DataLoader(dataset=mitdb_test,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__() 
        self.layer = nn.Sequential(
            nn.Conv1d(1, 3, 5),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(3, 5, 5),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(5, 10, 5),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(10, 10, 4),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(2, 2))
      
        self.fc_layer = nn.Sequential(
            nn.Linear(280, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
            )

    def forward(self, x):
        out = self.layer(x)
        
        out = out.view(10, -1)
        
        out = self.fc_layer(out)
        return out
    
model = CNN().to(device)

# 논문에는 정의 x
criterion = torch.nn.CrossEntropyLoss()    # 비용 함수에 소프트맥스 함수 포함되어져 있음.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 총 train 데이터 갯수: batch 수 *  batch_size
total_batch = len(data_loader)
print('총 배치의 수 : {}'.format(total_batch))

avg_loss = []
train_acc = []
test_acc = []



for epoch in range(training_epochs):
    avg_cost = 0
    accuracy = 0
    
    for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y는 레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.unsqueeze(1).float().to(device)
        Y = Y.long().to(device)
        
        optimizer.zero_grad()
        hypothesis = model(X)
                
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        prediction = model(X)
        correct_prediction = torch.argmax(prediction, 1) == Y

        accuracy += correct_prediction.float().sum()
        avg_cost += cost / total_batch

    test_accuracy = 0
    with torch.no_grad():
        for X_test, Y_test in data_loader_test:

            X_test = X_test.unsqueeze(1).float().to(device)
            Y_test = Y_test.long().to(device)

            prediction = model(X_test)
            correct_prediction = torch.argmax(prediction, 1) == Y_test
            test_accuracy += correct_prediction.float().sum()
    
    avg_loss.append(avg_cost)
    train_acc.append(accuracy.item()/ len(data_loader.dataset))
    test_acc.append(float(test_accuracy.item()/ len(data_loader_test.dataset)))

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
    print('train accuracy:', accuracy.item()/ len(data_loader.dataset))    
    print('test ccuracy:', test_accuracy.item()/ len(data_loader_test.dataset))
    
plt.rcParams["figure.figsize"] = (9.6,4)

plt.plot(train_acc)
plt.plot(test_acc)
plt.legend(['train accuracy','test accuracy'])
plt.show()

# 학습을 진행하지 않을 것이므로 torch.no_grad()
accuracy = 0
labels = []
guesses = []
with torch.no_grad():
    for X_test, Y_test in data_loader_test:

        X_test = X_test.unsqueeze(1).float().to(device)
        Y_test = Y_test.long().to(device)
        
        prediction = model(X_test)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        labels.extend(Y_test.tolist())
        guesses.extend(torch.argmax(prediction, 1).tolist())
        accuracy += correct_prediction.float().sum()        
print('Accuracy:', accuracy.item()/ len(data_loader_test.dataset))

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

for i in range(len(guesses)):
    #True Positives
    if labels[i] == 1 and guesses[i] == 1:
        true_positives += 1

    #True Negatives
    if labels[i] == 0 and guesses[i] == 0:
        true_negatives += 1
    #False Positives
    if labels[i] == 0 and guesses[i] == 1:
        false_positives += 1
    #False Negatives
    if labels[i] == 1 and guesses[i] == 0:
        false_negatives += 1
        
print('true_positives', true_positives)
print('true_negatives', true_negatives)
print('false_positives', false_positives)
print('false_negatives', false_negatives)

print('accuracy', accuracy_score(labels, guesses))
print('recall', recall_score(labels, guesses))
print('precision', precision_score(labels, guesses))
print('f1', f1_score(labels, guesses))

plt.rcParams["figure.figsize"] = (6,4)
plt.plot(np.arange(len(avg_loss)), avg_loss)
plt.title('train loss')