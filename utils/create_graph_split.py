import pandas as pd
import json
import re
import random
import os

test_user=['YTL', 'YSL', 'YSD', 'YRP']
val_user=['YRH', 'YRK']

labels=pd.read_csv("../../data/labels.csv")
directory="../../data/raw_EEG/"
train_label={}
test_label={}
valid_label={}



for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        id=filename[14:-4]
        row=labels.loc[labels['ID'] == int(id)]
        label=int(row['Label'])
        destination_folder=None
        if(row['User'].item() in val_user):
            destination_folder="valid"
            valid_label[id]=label
        elif(row['User'].item() in test_user):
            destination_folder="test"
            test_label[id]=label
        else:
            destination_folder="train"
            train_label[id]=label
        print(destination_folder)
#         os.rename("../../data/raw_EEG/"+"steller_graph_"+str(id)+".csv", "../../data/EEG/"+destination_folder+"/steller_graph_"+str(id)+".csv")

# with open("../../data/EEG/label/train_label.json", 'w') as f:
#     json.dump(train_label, f)
# with open("../../data/EEG/label/test_label.json", 'w') as f:
#     json.dump(test_label, f)
# with open("../../data/EEG/label/valid_label.json", 'w') as f:
#     json.dump(valid_label, f)