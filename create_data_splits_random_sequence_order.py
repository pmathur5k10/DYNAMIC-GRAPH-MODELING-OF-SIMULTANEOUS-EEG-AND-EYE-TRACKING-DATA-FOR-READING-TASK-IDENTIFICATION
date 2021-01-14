import pandas as pd
import json
import re
import random

df_test=[]
df_train=[]
df_val=[]

test_user=['YTL', 'YSL', 'YSD', 'YRP']
val_user=['YRH', 'YRK']

with open("../../data/word_level_data.json") as json_file:
    data = json.load(json_file)
    for row in data:
        text=""
        if(len(row['user_order_text'])==0):        
            text="Null "
        shuffled_list=row['user_order_text']
        random.shuffle(shuffled_list)        
        for i in shuffled_list:
            text=text+i+" "
        text = re.sub('[^A-Za-z0-9]+', ' ', text)
    
        if(row['user'] in test_user):
            df_test.append([text[:-1], int(row['label']), row['id'], row['user'], row['reading_type'], row['sentence']])
        elif(row['user'] in val_user):
            df_val.append([text[:-1], int(row['label']), row['id'], row['user'], row['reading_type'], row['sentence']])
        else:
            df_train.append([text[:-1], int(row['label']), row['id'], row['user'], row['reading_type'], row['sentence']])


df_train=pd.DataFrame(df_train, columns=['text', 'label', 'id', 'user', 'reading_type', 'sentence'])
df_test=pd.DataFrame(df_test, columns=['text', 'label', 'id', 'user', 'reading_type', 'sentence'])
df_val=pd.DataFrame(df_val, columns=['text', 'label', 'id', 'user', 'reading_type', 'sentence'])

# Write preprocessed data
df_train.to_csv("../../data/random_sequence_words" + '/train.csv', index=False)
df_val.to_csv("../../data/random_sequence_words" + '/valid.csv', index=False)
df_test.to_csv("../../data/random_sequence_words" + '/test.csv', index=False)
    
