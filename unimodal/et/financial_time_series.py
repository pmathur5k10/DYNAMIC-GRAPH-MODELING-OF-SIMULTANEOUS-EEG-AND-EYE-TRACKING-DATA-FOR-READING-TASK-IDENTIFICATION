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
import torch.nn.functional as F
from torchtrainer import TorchTrainer

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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
# device='cpu'

#Constants
SEQ_LEN=10
SEED = 1234
BATCH_SIZE=64
EPOCHS=25
INPUT_DIMENSION=SEQ_LEN
HIDDEN_DIMENSION=64
OUTPUT_DIMENSION=1
DROPOUT=0.2
BIDIRECTIONAL=True
NUM_LAYERS=2
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


class RNNInitEncoder(nn.Module):
    def __init__(self, embed_sizes, rnn_num_layers=1, input_feature_len=10, sequence_len=20, hidden_size=100, bidirectional=False, device='cpu'):
        super().__init__()
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.input_feature_len = input_feature_len
        self.num_layers = rnn_num_layers
        self.rnn_directions = 2 if bidirectional else 1
        self.embeds = nn.ModuleList([nn.Embedding(num_classes, output_size) for num_classes, output_size in embed_sizes])
        self.embed_to_ht = nn.Linear(sum([s[1] for s in embed_sizes]), self.hidden_size)
        self.gru = nn.GRU(
            num_layers = rnn_num_layers,
            input_size=input_feature_len,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.device = device

    def forward(self, input_seq, input_cat):
        embeds = [e(input_cat[:, i]) for i, e in enumerate(self.embeds)]
        embeds = torch.cat(embeds, 1)
        ht = self.embed_to_ht(embeds)
        ht.unsqueeze_(0)
        if (self.num_layers * self.rnn_directions) > 1:
            ht = ht.repeat(self.rnn_directions * self.num_layers, 1, 1)
        if input_seq.ndim < 3:
            input_seq.unsqueeze_(2)
        gru_out, hidden = self.gru(input_seq, ht)
        if self.rnn_directions > 1:
            gru_out = gru_out.view(input_seq.size(0), self.sequence_len, self.rnn_directions, self.hidden_size)
            gru_out = torch.sum(gru_out, axis=2)
        return gru_out, hidden.squeeze(0)


class RNNConcatEncoder(nn.Module):
    def __init__(self, embed_sizes, rnn_num_layers=1, input_feature_len=1, sequence_len=168, hidden_size=100, bidirectional=False, device='cpu'):
        super().__init__()
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.embeds = nn.ModuleList([nn.Embedding(num_classes, output_size) for num_classes, output_size in embed_sizes])
        self.input_feature_len = input_feature_len
        self.num_layers = rnn_num_layers
        self.rnn_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(
            num_layers = rnn_num_layers,
            input_size=input_feature_len,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.output_linear = nn.Linear(hidden_size + sum([s[1] for s in embed_sizes]), hidden_size)
        self.device = device

    def forward(self, input_seq, input_cat):
        embeds = [e(input_cat[:, i]) for i, e in enumerate(self.embeds)]
        embeds = torch.cat(embeds, 1)
        ht = torch.zeros(self.num_layers * self.rnn_directions, input_seq.size(0) , self.hidden_size, device=self.device)
        if input_seq.ndim < 3:
            input_seq.unsqueeze_(2)
        gru_out, hidden = self.gru(input_seq, ht)
        if self.rnn_directions > 1:
            gru_out = gru_out.view(input_seq.size(0), self.sequence_len, self.rnn_directions, self.hidden_size)
            gru_out = torch.sum(gru_out, axis=2)
        encoder_concat_hidden = self.output_linear(torch.cat((hidden.squeeze(0), embeds), axis=1))
        return gru_out, encoder_concat_hidden


#output shape
# bidirectional output is summed
# gru_out - (batch, sequence_len, hidden_size)
# hidden - (batch, hidden_size) only the last layer for multi-layer
class RNNEncoder(nn.Module):
    def __init__(self, rnn_num_layers=1, input_feature_len=10, sequence_len=20, hidden_size=100, bidirectional=False, device=device, rnn_dropout=0.2):
        super().__init__()
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.input_feature_len = input_feature_len
        self.num_layers = rnn_num_layers
        self.rnn_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(
            num_layers=rnn_num_layers,
            input_size=input_feature_len,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=rnn_dropout
        )
        self.device = device

    def forward(self, input_seq):
        ht = torch.zeros(self.num_layers * self.rnn_directions, input_seq.size(0), self.hidden_size, device=self.device)
        if input_seq.ndim < 3:
            input_seq.unsqueeze_(2)
        gru_out, hidden = self.gru(input_seq, ht)
        print(gru_out.shape)
        print(hidden.shape)
        if self.rnn_directions * self.num_layers > 1:
            num_layers = self.rnn_directions * self.num_layers
            if self.rnn_directions > 1:
                gru_out = gru_out.view(input_seq.size(0), self.sequence_len, self.rnn_directions, self.hidden_size)
                gru_out = torch.sum(gru_out, axis=2)
            hidden = hidden.view(self.num_layers, self.rnn_directions, input_seq.size(0), self.hidden_size)
            if self.num_layers > 0:
                hidden = hidden[-1]
            else:
                hidden = hidden.squeeze(0)
            hidden = hidden.sum(axis=0)
        else:
            hidden.squeeze_(0)
        return gru_out, hidden

class DecoderCell(nn.Module):
    def __init__(self, input_feature_len, hidden_size, dropout=0.2):
        super().__init__()
        self.decoder_rnn_cell = nn.GRUCell(
            input_size=input_feature_len,
            hidden_size=hidden_size,
        )
        self.out = nn.Linear(hidden_size, 1)
        self.attention = False
        self.dropout = nn.Dropout(dropout)

    def forward(self, prev_hidden, y):
        rnn_hidden = self.decoder_rnn_cell(y, prev_hidden)
        output = self.out(rnn_hidden)
        return output, self.dropout(rnn_hidden)


class AttentionDecoderCell(nn.Module):
    def __init__(self, input_feature_len, hidden_size, sequence_len, dropout=0.2):
        super().__init__()
        # attention - inputs - (decoder_inputs, prev_hidden)
        self.attention_linear = nn.Linear(hidden_size + input_feature_len, sequence_len)
        self.attention = True
        # attention_combine - inputs - (decoder_inputs, attention * encoder_outputs)
        self.decoder_rnn_cell = nn.GRUCell(
            input_size=hidden_size,
            hidden_size=hidden_size,
    )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, encoder_output, prev_hidden, y):
        attention_input = torch.cat((prev_hidden, y), axis=1)
        attention_weights = F.softmax(self.attention_linear(attention_input)).unsqueeze(1)
        attention_combine = torch.bmm(attention_weights, encoder_output).squeeze(1)
        rnn_hidden = self.decoder_rnn_cell(attention_combine, prev_hidden)
        output = self.out(rnn_hidden)
        return output, self.dropout(rnn_hidden)

class EncoderDecoderWrapper(nn.Module):
    def __init__(self, encoder, decoder_cell, output_size=3, teacher_forcing=0.3, sequence_len=336, decoder_input=True, device=device):
        super().__init__()
        self.encoder = encoder
        self.decoder_cell = decoder_cell
        self.output_size = output_size
        self.teacher_forcing = teacher_forcing
        self.sequence_length = sequence_len
        self.decoder_input = decoder_input
        self.device = device

    def forward(self, xb, yb=None):
        if self.decoder_input:
            print(xb.shape)
            decoder_input = xb[-1]
            input_seq = xb[0]
            if len(xb) > 2:
                encoder_output, encoder_hidden = self.encoder(input_seq, *xb[1:-1])
            else:
                encoder_output, encoder_hidden = self.encoder(input_seq)
        else:
            if type(xb) is list and len(xb) > 1:
                input_seq = xb[0]
                encoder_output, encoder_hidden = self.encoder(*xb)
            else:
                input_seq = xb
                encoder_output, encoder_hidden = self.encoder(input_seq)
        prev_hidden = encoder_hidden
        outputs = torch.zeros(input_seq.size(0), self.output_size, device=self.device)
        y_prev = input_seq[:, -1, 0].unsqueeze(1)
        for i in range(self.output_size):
            step_decoder_input = torch.cat((y_prev, decoder_input[:, i]), axis=1)
            if (yb is not None) and (i > 0) and (torch.rand(1) < self.teacher_forcing):
                step_decoder_input = torch.cat((yb[:, i].unsqueeze(1), decoder_input[:, i]), axis=1)
            rnn_output, prev_hidden = self.decoder_cell(prev_hidden, step_decoder_input)
            y_prev = rnn_output
            outputs[:, i] = rnn_output.squeeze(1)
        return outputs

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


encoder = RNNEncoder(
    input_feature_len=71, 
    rnn_num_layers=1, 
    hidden_size=100,  
    sequence_len=180,
    bidirectional=False,
    device=device,
    rnn_dropout=0.2
)

decoder_cell = DecoderCell(
    input_feature_len=10,
    hidden_size=100,
)

loss_function=nn.BCEWithLogitsLoss()
loss_function=loss_function.to(device)

encoder = encoder.to(device)
decoder_cell = decoder_cell.to(device)
model = EncoderDecoderWrapper(
    encoder,
    decoder_cell,
    output_size=90,
    teacher_forcing=0,
    sequence_len=20,
    decoder_input=True,
    device=device
)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)

# encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-2)
# decoder_optimizer = torch.optim.AdamW(decoder_cell.parameters(), lr=1e-3, weight_decay=1e-2)

# encoder_scheduler = optim.lr_scheduler.OneCycleLR(encoder_optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=6)
# decoder_scheduler = optim.lr_scheduler.OneCycleLR(decoder_optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=6)


# trainer = TorchTrainer(model)
# trainer.prepare(model_optimizer,
#             loss_function,
#             train_loader,
#             valid_loader,)


# trainer.train(EPOCHS, BATCH_SIZE)

# trainer.train(6, train_loader, valid_loader, resume_only_model=True, resume=True)

# model=timeNet(NUM_LAYERS, INPUT_DIMENSION, HIDDEN_DIMENSION, NUM_CLASSES, DROPOUT)
print(model)
print(model.parameters())
# optimizer=optim.Adam(model.parameters(), lr=0.1)
# criterion=nn.BCEWithLogitsLoss()
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr= 0.0001)
# criterion = criterion.to(device)

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

    train_loss, train_acc=trainModel(model, train_loader, optimizer, loss_function)
    valid_loss, valid_acc=evaluate(model, valid_loader, optimizer, loss_function)

    if(valid_loss < best_valid_loss):
        best_valid_loss=valid_loss
        torch.save(model.state_dict(), PATH)
  
    print("Epoch: ", epoch, "train_loss: ", train_loss, "valid_loss: ", valid_loss, "train_acc: ", train_acc, "valid_acc: ", valid_acc)

model.load_state_dict(torch.load(PATH))

test_loss, test_acc=evaluate(model, test_loader, optimizer, loss_function)
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

