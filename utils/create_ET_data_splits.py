import pandas as pd
import json
import re
import random
import os


df_test=[]
df_train=[]
df_val=[]

df_train_labels=[]
df_test_labels=[]
df_valid_labels=[]

test_user=['YTL', 'YSL', 'YSD', 'YRP']
val_user=['YRH', 'YRK']
directory="../../data/ET"
N=100
lmin=1000
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        with open(os.path.join(directory, filename), 'r') as json_file:

            row = json.load(json_file)
            temp=[]
            temp.append(row['FFD'])
            # temp.append(row['SFD'])
            temp.append(row['GD'])
            # temp.append(row['GPT'])
            temp.append(row['TRT'])
            temp.append(row['FFD_pupilsize'])
            # temp.append(row['SFD_pupilsize'])
            temp.append(row['GD_pupilsize'])
            # temp.append(row['GPT_pupilsize'])
            temp.append(row['TRT_pupilsize'])
            if(row['user'] in test_user):
                df_test.extend(temp)
                df_test_labels.extend(row['label'])
            elif(row['user'] in val_user):
                df_val.extend(temp)
                df_valid_labels.extend(row['label'])
            else:
                df_train.extend(temp)
                df_train_labels.extend(row['label'])
            
df_train_padded=[]
for line in df_train:
    if(len(line)>=N):
        df_train_padded.append(line[:N])
    else:
        df_train_padded.append(line + ['0.01'] * (N - len(line)+1))

df_val_padded=[]
for line in df_val:
    if(len(line)>=N):
        df_val_padded.append(line[:N])
    else:
        df_val_padded.append(line + ['0.01'] * (N - len(line)+1))

df_test_padded=[]
for line in df_test:
    if(len(line)>=N):
        df_test_padded.append(line[:N+1])
    else:
        df_test_padded.append(line + ['0.01'] * (N - len(line)+1))

df_train=pd.DataFrame(df_train_padded)
df_test=pd.DataFrame(df_test_padded)
df_val=pd.DataFrame(df_val_padded)

df_train_labels=pd.DataFrame(df_train_labels)
df_valid_labels=pd.DataFrame(df_valid_labels)
df_test_labels=pd.DataFrame(df_test_labels)



# Write preprocessed data
df_train.to_csv("../../data/ET/processed/train.csv", index=False)
df_val.to_csv("../../data/ET/processed/valid.csv", index=False)
df_test.to_csv("../../data/ET/processed/test.csv", index=False)

df_train_labels.to_csv("../../data/ET/processed/train_labels.csv", index=False)
df_valid_labels.to_csv("../../data/ET/processed/valid_labels.csv", index=False)
df_test_labels.to_csv("../../data/ET/processed/test_labels.csv", index=False)
