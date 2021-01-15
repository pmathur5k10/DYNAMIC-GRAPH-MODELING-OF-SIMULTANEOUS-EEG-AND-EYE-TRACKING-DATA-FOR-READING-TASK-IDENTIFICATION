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


torch.manual_seed(12345)


import pandas as pd
import numpy as np
import csv
import json
import pandas as pd
import numpy as np
import os


torch.cuda.empty_cache()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

DROPOUT=0.5
epochs = 10
learning_rate=0.001
BATCH_SIZE=8
hidden_channels=8
hidden_text=128
num_classes=2
num_node_features=1
N=100
feature_len_ET=10
input_dimension=6
num_layers=3
MAX_SIZE_VOCAB=3886
EMBEDDING_DIMENSION=100
BIDIRECTIONAL=True

# torch.cuda.set_device(1)
test_user=['YTL', 'YSL', 'YSD', 'YRP']
val_user=['YRH', 'YRK']
directory="../../data/"

node_features=[]
for i in range(0,105):
    node_features.append([0])
for i in range(105,210):
    node_features.append([1])
for i in range(210,315):
    node_features.append([2])
for i in range(315,420):
    node_features.append([3])
for i in range(420,525):
    node_features.append([4])
for i in range(525,630):
    node_features.append([5])
for i in range(630,735):
    node_features.append([6])
for i in range(735,840):
    node_features.append([7])

train_text_dict={}
text_text_dict={}
valid_text_dict={}

with open("../../data/text" + '/train.json', 'r') as f:
    train_text_dict=json.load(f)
with open("../../data/text" + '/test.json', 'r') as f:
    test_text_dict=json.load(f)
with open("../../data/text" + '/valid.json', 'r') as f:
    valid_text_dict=json.load(f)


train_text=[]
train_directory="../../data/EEG/train"
train_graphs=[]
train_id=[]
train_ET=[]
for filename in os.listdir(train_directory):
    if filename.endswith(".csv"):
        try:
            adj_matrix=pd.read_csv(os.path.join(train_directory, filename)).iloc[:,:2].values   
            id=filename[14:-4]
            temp=[]
            with open(directory+"/ET/ET_features_"+id+".json", 'r') as json_file:
                row = json.load(json_file)            
                temp.append(row['FFD'])
                temp.append(row['GD'])
                temp.append(row['TRT'])
                temp.append(row['FFD_pupilsize'])
                temp.append(row['GD_pupilsize'])
                temp.append(row['TRT_pupilsize'])
            et_features=[]
            for line in temp:
                if(len(line)>=N):
                    et_features.append(line[:N])
                else:
                    et_features.append(line + ['0.01'] * (N - len(line)+1))

            train_graphs.append(adj_matrix)
            train_id.append(id)
            train_ET.append(et_features)
            train_text.append([train_text_dict[id]])
        
        except:
            continue
        

test_text=[]
test_directory="../../data/EEG/test"
test_graphs=[]
test_id=[]
test_ET=[]
for filename in os.listdir(test_directory):
    if filename.endswith(".csv"):
        try:
            adj_matrix=pd.read_csv(os.path.join(test_directory, filename)).iloc[:,:2].values   
            id=filename[14:-4]            
            temp=[]
            with open(directory+"/ET/ET_features_"+id+".json", 'r') as json_file:
                row = json.load(json_file)            
                temp.append(row['FFD'])
                temp.append(row['GD'])
                temp.append(row['TRT'])
                temp.append(row['FFD_pupilsize'])
                temp.append(row['GD_pupilsize'])
                temp.append(row['TRT_pupilsize'])
            et_features=[]
            for line in temp:
                if(len(line)>=N):
                    et_features.append(line[:N])
                else:
                    et_features.append(line + ['0.01'] * (N - len(line)+1))

            test_graphs.append(adj_matrix)
            test_id.append(id)
            test_ET.append(et_features)
            test_text.append([test_text_dict[id]])
        
        except:
            continue


valid_text=[]
valid_directory="../../data/EEG/valid"
valid_graphs=[]
valid_id=[]
valid_ET=[]

for filename in os.listdir(valid_directory):
    if filename.endswith(".csv"):
        try:
            adj_matrix=pd.read_csv(os.path.join(valid_directory, filename)).iloc[:,:2].values   
            id=filename[14:-4]            
            temp=[]
            with open(directory+"/ET/ET_features_"+id+".json", 'r') as json_file:
                row = json.load(json_file)            
                temp.append(row['FFD'])
                temp.append(row['GD'])
                temp.append(row['TRT'])
                temp.append(row['FFD_pupilsize'])
                temp.append(row['GD_pupilsize'])
                temp.append(row['TRT_pupilsize'])
            et_features=[]
            for line in temp:
                if(len(line)>=N):
                    et_features.append(line[:N])
                else:
                    et_features.append(line + ['0.01'] * (N - len(line)+1))

            valid_graphs.append(adj_matrix)
            valid_id.append(id)
            valid_ET.append(et_features)
            valid_text.append([valid_text_dict[id]])
        except:
            continue



train_graph_labels={}
test_graph_labels={}
valid_graph_labels={}

train_labels=[]
test_labels=[]
valid_labels=[]

with open("../../data/EEG/label/train_label.json") as f:
    train_graph_labels=json.load(f)

with open("../../data/EEG/label/test_label.json") as f:
    test_graph_labels=json.load(f)

with open("../../data/EEG/label/valid_label.json") as f:
    valid_graph_labels=json.load(f)



for id in train_id:
    train_labels.append(int(train_graph_labels[id]))

for id in test_id:
    test_labels.append(int(test_graph_labels[id]))

for id in valid_id:
    valid_labels.append(int(valid_graph_labels[id]))

train_graph_dataset=[]
test_graph_dataset=[]
valid_graph_dataset=[]


for i in range(len(train_graphs)):
    edge_index = torch.tensor(train_graphs[i], dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float)
    y=torch.tensor(train_labels[i], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
    train_graph_dataset.append(data)
for i in range(len(test_graphs)):
    edge_index = torch.tensor(test_graphs[i], dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float)
    y=torch.tensor(test_labels[i], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
    test_graph_dataset.append(data)
for i in range(len(valid_graphs)):
    edge_index = torch.tensor(valid_graphs[i], dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float)
    y=torch.tensor(valid_labels[i], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
    valid_graph_dataset.append(data)

train_graph_loader = DataLoader(train_graph_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_graph_loader = DataLoader(test_graph_dataset, batch_size=BATCH_SIZE, shuffle=False)
valid_graph_loader=DataLoader(valid_graph_dataset, batch_size=BATCH_SIZE, shuffle=False)

TEXT=torchtext.data.Field(tokenize='spacy', include_lengths=True, sequential=True)
LABEL=torchtext.data.LabelField(dtype=torch.float, sequential=False, use_vocab=False)
fields = [('text', TEXT), ('label', LABEL)]


df_train_text=pd.DataFrame(train_text, columns=['text'])
df_test_text=pd.DataFrame(test_text, columns=['text'])
df_val_text=pd.DataFrame(valid_text, columns=['text'])

df_train_label=pd.DataFrame(train_labels, columns=['label'])
df_test_label=pd.DataFrame(test_labels, columns=['label'])
df_val_label=pd.DataFrame(valid_labels, columns=['label'])

df_train_text_label=pd.concat([df_train_text,df_train_label], axis=1)
df_test_text_label=pd.concat([df_test_text,df_test_label], axis=1)
df_val_text_label=pd.concat([df_val_text,df_val_label], axis=1)


# Write preprocessed data
df_train_text_label.to_csv("../../data/user_seq_text_label" + '/train.csv', index=False)
df_val_text_label.to_csv("../../data/user_seq_text_label" + '/valid.csv', index=False)
df_test_text_label.to_csv("../../data/user_seq_text_label" + '/test.csv', index=False)
  

train_text_dataset, val_text_dataset, test_text_dataset= TabularDataset.splits(path="../../data/user_seq_text_label", train='train.csv', validation='valid.csv', test='test.csv',
                                           format='CSV', fields=fields, skip_header=True)


TEXT.build_vocab(train_text_dataset, max_size= MAX_SIZE_VOCAB, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_text_dataset)

train_text_loader=torchtext.data.Iterator(train_text_dataset, batch_size=BATCH_SIZE, device=device)
test_text_loader=torchtext.data.Iterator(test_text_dataset, batch_size=BATCH_SIZE, device=device)
valid_text_loader=torchtext.data.Iterator(val_text_dataset, batch_size=BATCH_SIZE, device=device)



class TimeseriesDataset(torch.utils.data.Dataset):   
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return int(np.ceil(len(self.X)))

    def __getitem__(self, index):

        data=self.X[index]
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j]=float(data[i][j])        
        data=np.array(data)        
        data=data[:,:feature_len_ET]
        data=np.transpose(data)
        label=np.array(float(self.y[index]))
        return (torch.tensor(data), torch.tensor(label))

train_ET_dataset = TimeseriesDataset(train_ET, train_labels)
train_ET_loader = torch.utils.data.DataLoader(train_ET_dataset, batch_size = BATCH_SIZE, shuffle = False)

test_ET_dataset = TimeseriesDataset(test_ET, test_labels)
test_ET_loader = torch.utils.data.DataLoader(test_ET_dataset, batch_size = BATCH_SIZE, shuffle = False)

valid_ET_dataset = TimeseriesDataset(valid_ET, valid_labels)
valid_ET_loader = torch.utils.data.DataLoader(valid_ET_dataset, batch_size = BATCH_SIZE, shuffle = False)

# class TimeBlock(nn.Module):
#     """
#     Neural network block that applies a temporal convolution to each node of
#     a graph in isolation.
#     """

#     def __init__(self, in_channels, out_channels, kernel_size=3):
#         """
#         :param in_channels: Number of input features at each node in each time
#         step.
#         :param out_channels: Desired number of output channels at each node in
#         each time step.
#         :param kernel_size: Size of the 1D temporal kernel.
#         """
#         super(TimeBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
#         self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
#         self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

#     def forward(self, X):
#         """
#         :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
#         num_features=in_channels)
#         :return: Output data of shape (batch_size, num_nodes,
#         num_timesteps_out, num_features_out=out_channels)
#         """
#         # Convert into NCHW format for pytorch to perform convolutions.
#         X = X.permute(0, 3, 1, 2)
#         temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
#         out = F.relu(temp + self.conv3(X))
#         # Convert back from NCHW to NHWC
#         out = out.permute(0, 2, 3, 1)
#         return out


# class STGCNBlock(nn.Module):
#     """
#     Neural network block that applies a temporal convolution on each node in
#     isolation, followed by a graph convolution, followed by another temporal
#     convolution on each node.
#     """

#     def __init__(self, in_channels, spatial_channels, out_channels,
#                  num_nodes):
#         """
#         :param in_channels: Number of input features at each node in each time
#         step.
#         :param spatial_channels: Number of output channels of the graph
#         convolutional, spatial sub-block.
#         :param out_channels: Desired number of output features at each node in
#         each time step.
#         :param num_nodes: Number of nodes in the graph.
#         """
#         super(STGCNBlock, self).__init__()
#         self.temporal1 = TimeBlock(in_channels=in_channels,
#                                    out_channels=out_channels)
#         self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
#                                                      spatial_channels))
#         self.temporal2 = TimeBlock(in_channels=spatial_channels,
#                                    out_channels=out_channels)
#         self.batch_norm = nn.BatchNorm2d(num_nodes)
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.Theta1.shape[1])
#         self.Theta1.data.uniform_(-stdv, stdv)

#     def forward(self, X, A_hat):
#         """
#         :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
#         num_features=in_channels).
#         :param A_hat: Normalized adjacency matrix.
#         :return: Output data of shape (batch_size, num_nodes,
#         num_timesteps_out, num_features=out_channels).
#         """
#         t = self.temporal1(X)
#         lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
#         # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
#         t2 = F.relu(torch.matmul(lfs, self.Theta1))
#         t3 = self.temporal2(t2)
#         return self.batch_norm(t3)
#         # return t3


# class STGCN(nn.Module):
#     """
#     Spatio-temporal graph convolutional network as described in
#     https://arxiv.org/abs/1709.04875v3 by Yu et al.
#     Input should have shape (batch_size, num_nodes, num_input_time_steps,
#     num_features).
#     """

#     def __init__(self, num_nodes, num_features, num_timesteps_input,
#                  num_timesteps_output):
#         """
#         :param num_nodes: Number of nodes in the graph.
#         :param num_features: Number of features at each node in each time step.
#         :param num_timesteps_input: Number of past time steps fed into the
#         network.
#         :param num_timesteps_output: Desired number of future time steps
#         output by the network.
#         """
#         super(STGCN, self).__init__()
#         self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
#                                  spatial_channels=16, num_nodes=num_nodes)
#         self.block2 = STGCNBlock(in_channels=64, out_channels=64,
#                                  spatial_channels=16, num_nodes=num_nodes)
#         self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
#         self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64,
#                                num_timesteps_output)

#     def forward(self, A_hat, X):
#         """
#         :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
#         num_features=in_channels).
#         :param A_hat: Normalized adjacency matrix.
#         """
#         out1 = self.block1(X, A_hat)
#         out2 = self.block2(out1, A_hat)
#         out3 = self.last_temporal(out2)
#         out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
#         return out4

# class timeNet(nn.Module):
#     '''
#     model for timeseries classification
#     '''
#     def __init__(self, num_layers, input_size, hidden_size, num_classes, dropout):
#         super(timeNet, self).__init__()
#         self.lstm= nn.LSTM(input_size,hidden_size,num_layers)
#         self.linear = nn.Linear(hidden_size, num_classes)
#         self.dropout=nn.Dropout(dropout)

#     def init_weights(self):
#         self.linear.weight.data.uniform_(-0.1,0.1)
#         self.linear.bias.data.fill_(4)

#     def forward(self,batch_input):
#         out,_ = self.lstm(batch_input)
#         out = self.linear(out[:,-1, :])
#         return out


PAD_IDX=TEXT.vocab.stoi[TEXT.pad_token]


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        self.fc_out1 = nn.Linear(2*hidden_channels, hidden_channels)
        self.fc_out2= nn.Linear(2*hidden_channels, hidden_channels)
        self.fc_final = nn.Linear(2*hidden_channels, num_classes)

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        self.lstm= nn.LSTM(input_dimension,hidden_channels,num_layers)
        self.linear = nn.Linear(hidden_channels, hidden_channels)


        self.embedding=nn.Embedding(MAX_SIZE_VOCAB, EMBEDDING_DIMENSION, padding_idx=PAD_IDX)
        self.rnn=nn.LSTM(EMBEDDING_DIMENSION, hidden_channels,num_layers=num_layers, bidirectional=BIDIRECTIONAL, dropout=DROPOUT)
        self.fc=nn.Linear(hidden_channels * 2, hidden_channels)

    def forward(self, x, edge_index, batch, x2, text, text_lenghts):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x1=F.dropout(x, p=DROPOUT, training=self.training)
        x1 = x1.view(x1.size(0), -1)
        # print(x1.shape)

        out,_ = self.lstm(x2)
        x2 = self.linear(out[:,-1, :])
        x2=F.dropout(x2, p=DROPOUT)
        x2 = x2.view(x2.size(0), -1)
        # print(x2.shape)

        x_prime = torch.cat((x1, x2), 1)
        x_prime = self.fc_out1(x_prime)
        x_prime=F.dropout(x_prime, p=DROPOUT)
        x_prime = x_prime.view(x_prime.size(0), -1)

        embedded=F.dropout(self.embedding(text))
        packed_embedding=nn.utils.rnn.pack_padded_sequence(embedded, text_lenghts, enforce_sorted = False)
        output, (hidden, cell)=self.rnn(packed_embedding)
        hidden=F.dropout(torch.cat((hidden[-1,:,:], hidden[-2,:,:]), dim=1), p=DROPOUT)
        x3=self.fc(hidden)
        x3 = x3.view(x3.size(0), -1)
        # print(x3.shape)

        # Concatenate in dim1 (feature dimension)
        x_double_prime = torch.cat((x2, x3), 1)
        x_double_prime=self.fc_out2(x_double_prime)
        x_double_prime=F.dropout(x_double_prime, p=DROPOUT)
        x_double_prime = x_double_prime.view(x_double_prime.size(0), -1)
        # print(x.shape)

        x=torch.cat((x_prime, x_double_prime), 1)
        x = self.fc_final(x)
        return x


model = MyModel().to(device)
model=model.float()

pre_trained_embedding=TEXT.vocab.vectors
model.embedding.weight.data.copy_(pre_trained_embedding)
UNK_IDX=TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[UNK_IDX]=torch.zeros(EMBEDDING_DIMENSION)
model.embedding.weight.data[PAD_IDX]=torch.zeros(EMBEDDING_DIMENSION)

print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss().to(device)

def train():
    model.train()

    for data in zip(train_graph_loader, train_ET_loader, train_text_loader):  # Iterate in batches over the training dataset.
        out = model(data[0].x.to(device), data[0].edge_index.to(device), data[0].batch.to(device), data[1][0].float().to(device), data[2].text[0], data[2].text[1])  # Perform a single forward pass.
        loss = criterion(out, data[0].y.to(device))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(graph_loader, ET_loader, text_loader):
    model.eval()

    correct = 0
    epoch_loss=0.0
    epoch_acc=0.0
    for data in zip(graph_loader, ET_loader, text_loader):  # Iterate in batches over the training/test dataset.
        out = model(data[0].x.to(device), data[0].edge_index.to(device), data[0].batch.to(device), data[1][0].float().to(device), data[2].text[0], data[2].text[1])  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        loss = criterion(out, data[0].y.to(device))
        epoch_loss+=loss
        correct += int((pred == data[0].y.to(device)).sum())  # Check against ground-truth labels.
    return correct / len(graph_loader.dataset), epoch_loss  # Derive ratio of correct predictions.


for epoch in range(1, epochs):
    train()
    train_acc = test(train_graph_loader, train_ET_loader, train_text_loader)
    val_acc, val_loss = test(valid_graph_loader, valid_ET_loader, valid_text_loader)
    print("Epoch:",epoch, "Train Acc:", train_acc, "Val Loss:", val_loss, "Val Acc:", val_acc)

print(test(test_graph_loader, test_ET_loader, test_text_loader))