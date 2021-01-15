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


torch.manual_seed(12345)


import pandas as pd
import numpy as np
import csv
import json
import pandas as pd
import numpy as np
import os


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

DROPOUT=0.5
epochs = 10
learning_rate=0.01
BATCH_SIZE=64
pooling_ratio=0.5
hidden_channels=256
num_classes=2
num_node_features=1
N=100
feature_len_ET=10
input_dimension=6
num_layers=1
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
        
        except:
            continue
        

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
        
        except:
            continue


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

#     def __init__(self, num_nodes, num_features, num_timesteps_input):
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

from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch.nn import Parameter


class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.tanh):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.score_layer(x,edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        self.fc_out = nn.Linear(2*hidden_channels, num_classes)

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.pool1 = SAGPool(hidden_channels, ratio=pooling_ratio)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool2 = SAGPool(hidden_channels, ratio=pooling_ratio)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        self.lstm= nn.LSTM(input_dimension,hidden_channels,num_layers)
        self.linear = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, batch, x2):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        # x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x1=F.dropout(x, p=DROPOUT, training=self.training)
        x1 = x1.view(x1.size(0), -1)
        
        out,_ = self.lstm(x2)
        x2 = self.linear(out[:,-1, :])
        x2=F.dropout(x2, p=DROPOUT, training=self.training)
        x2 = x2.view(x2.size(0), -1)

        # Concatenate in dim1 (feature dimension)
        x = torch.cat((x1, x2), 1)
        x = self.fc_out(x)
        return x


# num_nodes=840
# num_features=1
# num_timesteps_input=feature_len_ET
# stgcn_model=STGCN(num_nodes, num_features, num_timesteps_input)
# model=stgcn_model.float()
# print(model)

model = MyModel()
model=model.float()
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in zip(train_graph_loader, train_ET_loader):  # Iterate in batches over the training dataset.
        out = model(data[0].x, data[0].edge_index, data[0].batch, data[1][0].float())  # Perform a single forward pass.
        loss = criterion(out, data[0].y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(graph_loader, ET_loader):
    model.eval()

    correct = 0
    epoch_loss=0.0
    epoch_acc=0.0
    for data in zip(graph_loader, ET_loader):  # Iterate in batches over the training/test dataset.
        out = model(data[0].x, data[0].edge_index, data[0].batch, data[1][0].float())  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        loss = criterion(out, data[0].y)
        epoch_loss+=loss
        correct += int((pred == data[0].y).sum())  # Check against ground-truth labels.
    return correct / len(graph_loader.dataset), epoch_loss  # Derive ratio of correct predictions.


for epoch in range(1, epochs):
    train()
    train_acc = test(train_graph_loader, train_ET_loader)
    val_acc, val_loss = test(valid_graph_loader, valid_ET_loader)
    print("Epoch:",epoch, "Train Acc:", train_acc, "Val Loss:", val_loss, "Val Acc:", val_acc)

print(test(test_graph_loader, test_ET_loader))