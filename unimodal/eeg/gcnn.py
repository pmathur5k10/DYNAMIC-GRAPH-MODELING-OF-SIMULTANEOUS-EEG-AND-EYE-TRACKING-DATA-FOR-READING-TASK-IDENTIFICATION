#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
import csv
import json
import pandas as pd
import numpy as np
import os

import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import GCNSupervisedGraphClassification
from stellargraph import StellarGraph

from stellargraph import datasets

from sklearn import model_selection
from IPython.display import display, HTML

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt


epochs = 300
learning_rate=100
batch_size=64

node_features=[]
for i in range(0,105):
    node_features.append([str(i), i, 0])
for i in range(105,210):
    node_features.append([str(i), i, 1])
for i in range(210,315):
    node_features.append([str(i), i, 2])
for i in range(315,420):
    node_features.append([str(i), i, 3])
for i in range(420,525):
    node_features.append([str(i), i, 4])
for i in range(525,630):
    node_features.append([str(i), i, 5])
for i in range(630,735):
    node_features.append([str(i), i, 6])
for i in range(735,840):
    node_features.append([str(i), i, 7])
        
df_node_features = pd.DataFrame(node_features, columns =['index', 'electrode', 'freq_band'])
    

train_directory="../../../data/EEG/train"
train_graphs=[]
train_graph_id=[]
for file in os.listdir(train_directory):
    if file.endswith(".csv"):
        adj_matrix=pd.read_csv(os.path.join(train_directory, file))
        graph = StellarGraph(nodes=df_node_features, edges=adj_matrix)        
        train_graphs.append(graph)
        train_graph_id.append(file[14:-4])


valid_directory="../../../data/EEG/valid"
valid_graphs=[]
valid_graph_id=[]
for file in os.listdir(valid_directory):
    if file.endswith(".csv"):
        adj_matrix=pd.read_csv(os.path.join(valid_directory, file))
        graph = StellarGraph(nodes=df_node_features, edges=adj_matrix)        
        valid_graphs.append(graph)
        valid_graph_id.append(file[14:-4])

test_directory="../../../data/EEG/test"
test_graphs=[]
test_graph_id=[]
for file in os.listdir(test_directory):
    if file.endswith(".csv"):
        adj_matrix=pd.read_csv(os.path.join(test_directory, file))
        graph = StellarGraph(nodes=df_node_features, edges=adj_matrix)        
        test_graphs.append(graph)
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



train_graph_labels=pd.DataFrame(data=train_graph_labels, columns=['label'])
test_graph_labels=pd.DataFrame(data=test_graph_labels, columns=['label'])
valid_graph_labels=pd.DataFrame(data=valid_graph_labels, columns=['label'])

def create_graph_classification_model(generator):
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[64, 64],
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.5,
    )
    x_inp, x_out = gc_model.in_out_tensors()
    predictions = Dense(units=32, activation="relu")(x_out)
    # predictions = Dense(units=16, activation="relu")(predictions)
    predictions = Dense(units=1, activation="sigmoid")(predictions)

    # Let's create the Keras model and prepare it for training
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate), loss=binary_crossentropy, metrics=["acc"])

    return model


es = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True
)



def trainModel(model, train_generator, valid_generator, es, epochs):
    history = model.fit(
        train_generator, epochs=epochs, validation_data=valid_generator, verbose=2
    )
    # calculate performance on the test data and return along with history
    test_metrics = model.evaluate(test_generator, verbose=2)
    test_acc = test_metrics[model.metrics_names.index("acc")]

    return history, test_acc

train_gen = PaddedGraphGenerator(graphs=train_graphs)
train_generator=train_gen.flow(graphs=train_graphs, targets=train_graph_labels.values, batch_size=batch_size)
valid_gen = PaddedGraphGenerator(graphs=valid_graphs)
valid_generator=valid_gen.flow(graphs=valid_graphs, targets=valid_graph_labels.values, batch_size=batch_size)
test_gen = PaddedGraphGenerator(graphs=test_graphs)
test_generator=test_gen.flow(graphs=test_graphs, targets=test_graph_labels.values, batch_size=batch_size)

print("begin training")
model = create_graph_classification_model(train_gen)
trainModel(model, train_generator, valid_generator, es, epochs)
history, test_acc=model.evaluate(test_generator)
print(test_acc)