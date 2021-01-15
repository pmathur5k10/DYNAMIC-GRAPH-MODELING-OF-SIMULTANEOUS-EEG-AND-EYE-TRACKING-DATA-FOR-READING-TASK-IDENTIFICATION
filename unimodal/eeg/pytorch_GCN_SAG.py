import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

torch.manual_seed(12345)

from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch.nn import Parameter


import pandas as pd
import numpy as np
import csv
import json
import pandas as pd
import numpy as np
import os


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

epochs = 100
learning_rate=0.001
batch_size=1
hidden_channels=64
num_classes=2
num_node_features=1
pooling_ratio=0.5

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


train_directory="../../../data/EEG/train"
train_graphs=[]
train_graph_id=[]
for file in os.listdir(train_directory):
    if file.endswith(".csv"):
        adj_matrix=pd.read_csv(os.path.join(train_directory, file)).iloc[:,:2].values   
        # graph = StellarGraph(nodes=df_node_features, edges=adj_matrix)        
        train_graphs.append(adj_matrix)
        train_graph_id.append(file[14:-4])


valid_directory="../../../data/EEG/valid"
valid_graphs=[]
valid_graph_id=[]
for file in os.listdir(valid_directory):
    if file.endswith(".csv"):
        adj_matrix=pd.read_csv(os.path.join(valid_directory, file)).iloc[:,:2].values
        # graph = StellarGraph(nodes=df_node_features, edges=adj_matrix)        
        valid_graphs.append(adj_matrix)
        valid_graph_id.append(file[14:-4])

test_directory="../../../data/EEG/test"
test_graphs=[]
test_graph_id=[]
for file in os.listdir(test_directory):
    if file.endswith(".csv"):
        adj_matrix=pd.read_csv(os.path.join(test_directory, file)).iloc[:,:2].values
        # graph = StellarGraph(nodes=df_node_features, edges=adj_matrix)        
        test_graphs.append(adj_matrix)
        test_graph_id.append(file[14:-4])



train_graph_labels=[]
test_graph_labels=[]
valid_graph_labels=[]

train_labels={}
test_labels={}
valid_labels={}

with open("../../../data/EEG/label/train_label.json") as f:
    train_labels=json.load(f)

with open("../../../data/EEG/label/test_label.json") as f:
    test_labels=json.load(f)

with open("../../../data/EEG/label/valid_label.json") as f:
    valid_labels=json.load(f)



for id in train_graph_id:
    train_graph_labels.append(int(train_labels[id]))

for id in test_graph_id:
    test_graph_labels.append(int(test_labels[id]))

for id in valid_graph_id:
    valid_graph_labels.append(int(valid_labels[id]))


train_dataset=[]
test_dataset=[]
valid_dataset=[]

for i in range(len(train_graphs)):
    edge_index = torch.tensor(train_graphs[i], dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float)
    y=torch.tensor(train_graph_labels[i], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
    train_dataset.append(data)
for i in range(len(test_graphs)):
    edge_index = torch.tensor(test_graphs[i], dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float)
    y=torch.tensor(test_graph_labels[i], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
    test_dataset.append(data)
for i in range(len(valid_graphs)):
    edge_index = torch.tensor(valid_graphs[i], dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float)
    y=torch.tensor(valid_graph_labels[i], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
    valid_dataset.append(data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
valid_loader=DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


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



class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.pool1 = SAGPool(hidden_channels, ratio=pooling_ratio)
        self.conv2 = GCNConv(hidden_channels, int(hidden_channels/2))
        self.pool2 = SAGPool(int(hidden_channels/2), ratio=pooling_ratio)
        self.conv3 = GCNConv(int(hidden_channels/2), int(hidden_channels/4))
        self.lin = Linear(int(hidden_channels/4), num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

model = GCN()
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
    model.eval()

    correct = 0
    epoch_loss=0.0
    epoch_acc=0.0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        loss = criterion(out, pred)
        epoch_loss+=loss
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset), epoch_loss  # Derive ratio of correct predictions.


for epoch in range(1, epochs):
    train()
    train_acc = test(train_loader)
    val_acc, val_loss = test(valid_loader)
    print("Epoch:",epoch, "Train Acc:", train_acc, "Val Loss:", val_loss, "Val Acc:", val_acc)

print(test(test_loader))