import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from net import gtnet

import torch
import torchtext
from torchtext import data
import random
import spacy

from sklearn.model_selection import train_test_split
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import seaborn as sns
from prettytable import PrettyTable


torch.manual_seed(12345)


import pandas as pd
import numpy as np
import csv
import json
import pandas as pd
import numpy as np
import os


BATCH_SIZE=1
learning_rate=1
epochs=10
DROPOUT=0.5


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

directory="../../../data/"

N=168
def pad_data(df):
    df=np.array(df)
    df=df.transpose()
    df=df.tolist()

    for i in range(len(df)):
        for j in range(len(df[i])):
            df[i][j]=float(df[i][j])

    df_new=[]

    for line in df:
        new_line=None
        if(len(line)>=N):
            new_line=line[:N]
        else:
            new_line=line + [min(line)] * (N - len(line))
        df_new.append(new_line)
    
    df_new=np.array(df_new)
    return df_new
    


train_directory="../../../data/EEG/train"
train_id=[]
train_ST_FFD=[]
train_ST_TRT=[]
train_ST_GPT=[]
train_ST_GD=[]
for filename in os.listdir(train_directory):
    if filename.endswith(".csv"):
        try:
            id=filename[14:-4]
            ST_FFD=[]
            # ST_TRT=[]
            # ST_GD=[]
            # ST_GPT=[]
            with open(directory+"/ST/ST_features_"+id+".json", 'r') as json_file:
                row = json.load(json_file)            
                ST_FFD=row['FFD']
                # ST_TRT=pad_data(row['GD'])
                # ST_GD=pad_data(row['GD'])
                # ST_GPT=pad_data(row['GPT'])
            
            if(len(ST_FFD)==0):
                continue
                      
            train_id.append(id)
            train_ST_FFD.append(ST_FFD)
            # train_ST_TRT.append(ST_TRT)
            # train_ST_GD.append(ST_GD)
            # train_ST_GPT.append(ST_GPT)
        
        except:
            continue
# print(len(train_ST_FFD))    

test_directory="../../../data/EEG/test"
test_id=[]
test_ST_FFD=[]
test_ST_TRT=[]
test_ST_GPT=[]
test_ST_GD=[]
for filename in os.listdir(test_directory):
    if filename.endswith(".csv"):
        try:
            id=filename[14:-4]
            ST_FFD=[]
            # ST_TRT=[]
            # ST_GD=[]
            # ST_GPT=[]
            with open(directory+"/ST/ST_features_"+id+".json", 'r') as json_file:
                row = json.load(json_file)            
                ST_FFD=row['FFD']
                # ST_TRT=pad_data(row['TRT'])
                # ST_GD=pad_data(row['GD'])
                # ST_GPT=pad_data(row['GPT'])
            
            if(len(ST_FFD)==0):
                continue
            
            test_id.append(id)
            test_ST_FFD.append(ST_FFD)
            # test_ST_TRT.append(ST_TRT)
            # test_ST_GD.append(ST_GD)
            # test_ST_GPT.append(ST_GPT)
        
        except:
            continue

# print(len(test_ST_FFD))

valid_directory="../../../data/EEG/valid"
valid_id=[]
valid_ST_FFD=[]
valid_ST_TRT=[]
valid_ST_GPT=[]
valid_ST_GD=[]
for filename in os.listdir(valid_directory):
    if filename.endswith(".csv"):
        try:
            id=filename[14:-4]
            ST_FFD=[]
            # ST_TRT=[]
            # ST_GD=[]
            # ST_GPT=[]
            with open(directory+"/ST/ST_features_"+id+".json", 'r') as json_file:
                row = json.load(json_file)            
                ST_FFD=row['FFD']
                # ST_TRT=pad_data(row['TRT'])
                # ST_GD=pad_data(row['GD'])
                # ST_GPT=pad_data(row['GPT'])
            
            if(len(ST_FFD)==0):
                continue

            valid_id.append(id)
            valid_ST_FFD.append(ST_FFD)
            # valid_ST_TRT.append(ST_TRT)
            # valid_ST_GD.append(ST_GD)
            # valid_ST_GPT.append(ST_GPT)
        
        except:
            continue


# print(len(valid_ST_FFD))

train_graph_labels={}
test_graph_labels={}
valid_graph_labels={}

train_labels=[]
test_labels=[]
valid_labels=[]

with open("../../../data/EEG/label/train_label.json") as f:
    train_graph_labels=json.load(f)

with open("../../../data/EEG/label/test_label.json") as f:
    test_graph_labels=json.load(f)

with open("../../../data/EEG/label/valid_label.json") as f:
    valid_graph_labels=json.load(f)



for id in train_id:
    train_labels.append(int(train_graph_labels[id]))

for id in test_id:
    test_labels.append(int(test_graph_labels[id]))

for id in valid_id:
    valid_labels.append(int(valid_graph_labels[id]))

# print(len(train_id), len(train_ST_FFD))
# exit()
class TimeseriesDataset(torch.utils.data.Dataset):   
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.feature_len_ST=168

    def __len__(self):
        return int(np.ceil(len(self.X)))

    def __getitem__(self, index):

        data=self.X[index]
        data=pad_data(data)
        label=np.array(float(self.y[index]))
        data=torch.tensor(data)
        data=torch.reshape(data, (1,data.shape[0], data.shape[1]))
        return (data, torch.tensor(label))

train_ST_FFD_dataset = TimeseriesDataset(train_ST_FFD, train_labels)
train_ST_FFD_loader = torch.utils.data.DataLoader(train_ST_FFD_dataset, batch_size = BATCH_SIZE, shuffle = False)

test_ST_FFD_dataset = TimeseriesDataset(test_ST_FFD, test_labels)
test_ST_FFD_loader = torch.utils.data.DataLoader(test_ST_FFD_dataset, batch_size = BATCH_SIZE, shuffle = False)

valid_ST_FFD_dataset = TimeseriesDataset(valid_ST_FFD, valid_labels)
valid_ST_FFD_loader = torch.utils.data.DataLoader(valid_ST_FFD_dataset, batch_size = BATCH_SIZE, shuffle = False)


model = gtnet(gcn_true=True, buildA_true=True, gcn_depth=2, num_nodes=840, device=device,dropout=DROPOUT, subgraph_size=105, node_dim=40, dilation_exponential=2, conv_channels=16, 
            residual_channels=16, skip_channels=32, end_channels=64, seq_length=168, in_dim=1, out_dim=1, 
            layers=1, propalpha=0.05, tanhalpha=3, layer_norm_affline=False)
model=model.float()

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model=model.to(device)
print(model)
# print(len(train_labels), sum(train_labels))

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    
# count_parameters(model)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.95)
criterion = torch.nn.BCEWithLogitsLoss().to(device)

def train(train_loader):
    model.train()
    i=0
    for data in train_loader:  # Iterate in batches over the training dataset.
        # print(i)
        # i=i+1
        out = model(data[0].float().to(device)).squeeze(1)  # Perform a single forward pass.
        loss = criterion(out, data[1].float().to(device))  # Compute the loss.
        loss.backward(retain_graph=False)  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        torch.cuda.empty_cache()

def test(ST_loader):
    

    with torch.no_grad():

        model.eval()

        correct = 0
        epoch_loss=0.0
        epoch_acc=0.0
        for data in ST_loader:  # Iterate in batches over the training/test dataset.
            out = model(data[0].float().to(device)).squeeze(1)
            pred = out.argmax(dims=1)  # Use the class with highest probability.
            loss = criterion(out, data[1].float().to(device))
            epoch_loss+=loss
            correct += int((pred == data[1].to(device)).sum())  # Check against ground-truth labels.
    return correct / len(ST_loader), epoch_loss  # Derive ratio of correct predictions.

def run(train_loader, valid_loader):
    print("begin training")
    for epoch in range(1, epochs):
        print("Epoch ", epoch)
        train(train_loader)
        train_acc = test(train_loader)
        val_acc, val_loss = test(valid_loader)
        print("Epoch:",epoch, "Train Acc:", train_acc, "Val Loss:", val_loss, "Val Acc:", val_acc)

run(train_ST_FFD_loader, valid_ST_FFD_loader)
print(test(test_ST_FFD_loader))

# y_pred = []
# y_true=[]

# for data in test_ST_FFD_loader:  # Iterate in batches over the training/test dataset.
#     y_true.extend(data[1].long().tolist())
#     y_pred.extend(torch.sigmoid(model(data[0].float().to(device)).view(-1)).cpu().tolist())
    
# # for i, batch in enumerate(test_iterator):
# #   y_true.extend(batch.label.cpu().tolist())
# #   y_pred.extend(torch.sigmoid(model(batch.text[0], batch.text[1]).view(-1)).cpu().tolist())
# print(len(y_pred), len(y_true))

# for i in range(len(y_pred)):
#   if(y_pred[i]>0.5):
#     y_pred[i]=1.0
#   else:
#     y_pred[i]=0.0

# print(f1_score(y_true, y_pred, average='micro'))
# print(classification_report(y_true, y_pred))
# print(confusion_matrix(y_true, y_pred))
