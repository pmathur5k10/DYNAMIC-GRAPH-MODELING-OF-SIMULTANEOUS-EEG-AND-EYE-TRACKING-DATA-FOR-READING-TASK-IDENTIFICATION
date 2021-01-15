import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import torch
import torchtext
from torchtext import data
import random
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import json
import spacy
from sklearn.model_selection import train_test_split
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import seaborn as sns

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
# device='cpu'

#Constants
SEQ_LEN=6
SEED = 1234
BATCH_SIZE=64
EPOCHS=25
INPUT_DIMENSION=SEQ_LEN
HIDDEN_DIMENSION=256
OUTPUT_DIMENSION=1
DROPOUT=0.5
BIDIRECTIONAL=True
NUM_LAYERS=5
NUM_CLASSES=1
MAX_SIZE_VOCAB=3886
PATH="./saved_models/best_model_user_et"

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
np.random.seed(SEED)
torch.cuda.set_device(0)  # if you have more than one CUDA device

X_train=pd.read_csv("../../../data/ET/processed/train.csv").values.tolist()
X_test=pd.read_csv("../../../data/ET/processed/test.csv").values.tolist()
X_valid=pd.read_csv("../../../data/ET/processed/valid.csv").values.tolist()

X_train_labels=pd.read_csv("../../../data/ET/processed/train_labels.csv").values.tolist()
X_test_labels=pd.read_csv("../../../data/ET/processed/test_labels.csv").values.tolist()
X_valid_labels=pd.read_csv("../../../data/ET/processed/valid_labels.csv").values.tolist()


class TimeseriesDataset(torch.utils.data.Dataset):   
    def __init__(self, X, y, seq_len=1):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        # self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return int(np.ceil(len(self.X)/self.seq_len))

    def __getitem__(self, index):
        return (np.transpose(np.array(self.X[index*self.seq_len:(index*self.seq_len)+self.seq_len])[:,:10]), np.array(self.y[index]))

train_dataset = TimeseriesDataset(X_train, X_train_labels, seq_len=SEQ_LEN)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = False)

test_dataset = TimeseriesDataset(X_test, X_test_labels, seq_len=SEQ_LEN)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)

valid_dataset = TimeseriesDataset(X_valid, X_valid_labels, seq_len=SEQ_LEN)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle = False)

class timeNet(nn.Module):
    '''
    model for timeseries classification
    '''
    def __init__(self, num_layers, input_size, hidden_size, num_classes, dropout):
        super(timeNet, self).__init__()
        self.lstm= nn.LSTM(input_size,hidden_size,num_layers)
        self.linear = nn.Linear(hidden_size, num_classes)
        self.dropout=nn.Dropout(dropout)

    def init_weights(self):
        self.linear.weight.data.uniform_(-0.1,0.1)
        self.linear.bias.data.fill_(4)

    def forward(self,batch_input):
        out,_ = self.lstm(batch_input)
        out = self.linear(out[:,-1, :])
        return out


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(10, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CustomModel(nn.Module):
  def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, vocab_size, dropout, bidirectional, n_layers, pad_idx):
    super().__init__()

    self.embedding=nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
    self.rnn=nn.LSTM(embedding_dim, hidden_dim,num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
    self.fc=nn.Linear(hidden_dim * 2, output_dim)
    self.dropout=nn.Dropout(dropout)

  def forward(self, text, text_lenghts):
    embedded=self.dropout(self.embedding(text))
    packed_embedding=nn.utils.rnn.pack_padded_sequence(embedded, text_lenghts)
    output, (hidden, cell)=self.rnn(packed_embedding)
    hidden=self.dropout(torch.cat((hidden[-1,:,:], hidden[-2,:,:]), dim=1))
    return self.fc(hidden)


model=timeNet(NUM_LAYERS, INPUT_DIMENSION, HIDDEN_DIMENSION, NUM_CLASSES, DROPOUT)
print(model)
print(model.parameters())
optimizer=optim.Adam(model.parameters(), lr=0.01)
criterion=nn.BCEWithLogitsLoss()
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr= 0.0001)

model = model.to(device)
criterion = criterion.to(device)

def accuracy(predicted, labels):

  rounded=torch.round(torch.sigmoid(predicted))
  correct=(rounded==labels).float()
  acc=correct.sum()/len(rounded)
  return acc

def trainModel(model, iterator, optimizer, criterion):

  epoch_loss=0.0
  epoch_acc=0.0

  model.train()

  for sample,label in iterator:

    sample=sample.to(device).float()
    label=label.to(device)
    optimizer.zero_grad()

    output=model(sample.float()).squeeze(1)
    label = label.squeeze(1).float()
    loss=criterion(output, label)

    acc=accuracy(output, label)
    loss.backward()
    optimizer.step()

    epoch_loss+=loss.item()
    epoch_acc+=acc.item()

  return epoch_loss/len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, optimizer, criterion):

  epoch_loss=0.0
  epoch_acc=0.0


  model.eval()

  with torch.no_grad():

    for sample,label in iterator:
        
        sample=sample.to(device).float()
        label=label.to(device)
        output=model(sample.float()).squeeze(1)
        label = label.squeeze(1).float()
        loss=criterion(output, label)
        acc=accuracy(output, label)
        
        epoch_loss+=loss.item()
        epoch_acc+=acc.item()
    
  # f1=f1_score(y_true, y_pred)
  return epoch_loss/len(iterator), epoch_acc / len(iterator)

best_valid_loss=float('inf')
for epoch in range(EPOCHS):

    train_loss, train_acc=trainModel(model, train_loader, optimizer, criterion)
    valid_loss, valid_acc=evaluate(model, valid_loader, optimizer, criterion)

    if(valid_loss < best_valid_loss):
        best_valid_loss=valid_loss
        torch.save(model.state_dict(), PATH)
  
    print("Epoch: ", epoch, "train_loss: ", train_loss, "valid_loss: ", valid_loss, "train_acc: ", train_acc, "valid_acc: ", valid_acc)

model.load_state_dict(torch.load(PATH))

test_loss, test_acc=evaluate(model, test_loader, optimizer, criterion)
print("test_loss: ", test_loss, "test_acc: ", test_acc*100)

# y_pred = []
# y_true=[]
# for sample, label in enumerate(test_loader):

#     print(sample, label)
#     y_true.extend(label.cpu())
#     y_pred.extend(torch.sigmoid(model(sample).view(-1)).cpu().tolist())

# for i in range(len(y_pred)):
#   if(y_pred[i]>0.5):
#     y_pred[i]=1.0
#   else:
#     y_pred[i]=0.0

# print(f1_score(y_true, y_pred, average='micro'))
# print(classification_report(y_true, y_pred))
# print(confusion_matrix(y_true, y_pred))
