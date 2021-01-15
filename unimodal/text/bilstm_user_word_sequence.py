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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

#Constants
SEED = 1234
BATCH_SIZE=128
EPOCHS=25
EMBEDDING_DIMENSION=100
HIDDEN_DIMENSION=256
OUTPUT_DIMENSION=1
DROPOUT=0.5
BIDIRECTIONAL=True
N_LAYERS=2
MAX_SIZE_VOCAB=3886
PATH="./saved_models/best_model_user"

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEXT=data.Field(tokenize='spacy', include_lengths=True, sequential=True)
LABEL=data.LabelField(dtype=torch.float, sequential=False, use_vocab=False)
fields = [('text', TEXT), ('label', LABEL)]

train, valid, test = TabularDataset.splits(path="../../../data/user_sequence_words", train='train.csv', validation='valid.csv', test='test.csv',
                                           format='CSV', fields=fields, skip_header=True)


TEXT.build_vocab(train, max_size= MAX_SIZE_VOCAB, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train)

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train, valid, test), batch_size=BATCH_SIZE, device=device, sort_within_batch=True, sort_key = lambda x: len(x.text))

class RNN(nn.Module):
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

PAD_IDX=TEXT.vocab.stoi[TEXT.pad_token]
INPUT_DIMENSION=len(TEXT.vocab)

model=RNN(INPUT_DIMENSION,EMBEDDING_DIMENSION, HIDDEN_DIMENSION, OUTPUT_DIMENSION, MAX_SIZE_VOCAB, DROPOUT, BIDIRECTIONAL, N_LAYERS, PAD_IDX)
print(model)

pre_trained_embedding=TEXT.vocab.vectors
print(pre_trained_embedding.shape)
model.embedding.weight.data.copy_(pre_trained_embedding)
UNK_IDX=TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[UNK_IDX]=torch.zeros(EMBEDDING_DIMENSION)
model.embedding.weight.data[PAD_IDX]=torch.zeros(EMBEDDING_DIMENSION)

optimizer=optim.Adam(model.parameters())
criterion=nn.BCEWithLogitsLoss()

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

  for batch in iterator:
    optimizer.zero_grad()
    text, text_lengths = batch.text
    output=model(text, text_lengths).squeeze(1)
    loss=criterion(output, batch.label)
    acc=accuracy(output, batch.label)
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

    for batch in iterator:
      text, text_lengths = batch.text
      output=model(text, text_lengths).squeeze(1)
      loss=criterion(output, batch.label)
      acc=accuracy(output, batch.label)
      
      epoch_loss+=loss.item()
      epoch_acc+=acc.item()
    
  # f1=f1_score(y_true, y_pred)
  return epoch_loss/len(iterator), epoch_acc / len(iterator)

best_valid_loss=float('inf')
for epoch in range(EPOCHS):

    train_loss, train_acc=trainModel(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc=evaluate(model, valid_iterator, optimizer, criterion)

    if(valid_loss < best_valid_loss):
        best_valid_loss=valid_loss
        torch.save(model.state_dict(), PATH)
  
    print("Epoch: ", epoch, "train_loss: ", train_loss, "valid_loss: ", valid_loss, "train_acc: ", train_acc, "valid_acc: ", valid_acc)

model.load_state_dict(torch.load(PATH))

test_loss, test_acc=evaluate(model, test_iterator, optimizer, criterion)
print("test_loss: ", test_loss, "test_acc: ", test_acc*100)

y_pred = []
y_true=[]
for i, batch in enumerate(test_iterator):
  y_true.extend(batch.label.cpu().tolist())
  y_pred.extend(torch.sigmoid(model(batch.text[0], batch.text[1]).view(-1)).cpu().tolist())

for i in range(len(y_pred)):
  if(y_pred[i]>0.5):
    y_pred[i]=1.0
  else:
    y_pred[i]=0.0

print(f1_score(y_true, y_pred, average='micro'))
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
