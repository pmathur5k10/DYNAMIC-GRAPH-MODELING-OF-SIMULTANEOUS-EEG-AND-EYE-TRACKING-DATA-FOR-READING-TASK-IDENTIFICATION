import pandas as pd
import json
import re

df_test={}
df_train={}
df_val={}

test_user=['YTL', 'YSL', 'YSD', 'YRP']
val_user=['YRH', 'YRK']

with open("../../data/word_level_data.json") as json_file:
    data = json.load(json_file)
    for row in data:
        text=""
        if(len(row['user_order_text'])==0):        
            text="Null "        
        for i in row['user_order_text']:
            text=text+i+" "
        text = re.sub('[^A-Za-z0-9]+', ' ', text)
    
        if(row['user'] in test_user):
            df_test[row['id']]=text[:-1]
        elif(row['user'] in val_user):
            df_val[row['id']]=text[:-1]
        else:
            df_train[row['id']]=text[:-1]
        print(text)



with open("../../data/text" + '/train.json', 'w') as f:
    json.dump(df_train, f)
with open("../../data/text" + '/test.json', 'w') as f:
    json.dump(df_test, f)
with open("../../data/text" + '/valid.json', 'w') as f:
    json.dump(df_val, f)
    
