#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding : utf-8 -*-
import pandas as pd
import numpy as np
import io 
import os 
import json 
import distutils.dir_util
from collections import Counter
from tensorflow.keras.models import save_model 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,LSTM,Dense, SimpleRNN,Dropout


# In[2]:


def load_train_json(fname):
    f = pd.read_json(fname+'.json' ,typ = 'frame', encoding="utf-8")
    df = pd.DataFrame(f)
    df = df.sort_values(by=['like_cnt'],ascending=False)
    df = df[df['like_cnt']>10]
    print('load_train_json')
    return df


# In[3]:


def load_val_json(fname):
    f = pd.read_json(fname+'.json' ,typ = 'frame', encoding="utf-8")
    df = pd.DataFrame(f)
    df = df.sort_values(by=['like_cnt'],ascending=False)
    #df = df[df['like_cnt']>10]
    print('load_train_json')
    return df


# In[4]:


def most_popular(df,song_num):
    train_song = df['songs']
    train_tag = df['tags']
    
    song_list = [song for plist in train_song for song in plist ]
    tag_list = [tag for plist in train_tag for tag in plist]
    
    count_song = Counter(song_list)
    count_tag = Counter(tag_list)
    x={}
    for key, value in count_song.items():
        if value>song_num:
            x[key]=value
    song_len = len(x)
    for key, value in count_tag.items():
        if value>1:
            x[key]=value
    tag_len=len(x)-song_len
    print('most_popular')
    return list(x.keys()),song_len, tag_len


# In[5]:


def create_zero(column_name):
    zero_df=pd.DataFrame(columns=col)
    return zero_df


# In[6]:


def create_onehot(df,column_name):
    zero_matrix=np.zeros((len(df),len(column_name)))
    zero_df=pd.DataFrame(zero_matrix,columns=column_name,index=df['id'])
    for i in range(len(df)):
        for tag,song in zip(df.iloc[i,0],df.iloc[i,3]):
            if tag in column_name:
                zero_df.iloc[i,column_name.index(tag)]=1
            if song in column_name:
                zero_df.iloc[i,column_name.index(song)]=1
    return zero_df
            


# In[7]:


def deep_learing(column_name,train_onehot,val_onehot):
        col = column_name
        
        encoding_dim=64
        input_plist=Input(shape=(len(col),))
        dropout = Dropout(0.2)(input_plist)
        encoded=Dense(encoding_dim,activation='relu')(input_plist)
        encoded=Dense(36,activation='relu')(encoded)
        decoded=Dense(len(col),activation='sigmoid')(encoded)
        autoencoder=Model(input_plist,decoded)
        
        autoencoder.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        
        autoencoder.fit(train_onehot,train_onehot,epochs=5,batch_size=64,validation_split=0.1)
        
        predict_plist=autoencoder.predict(val_onehot)
        print('deep_learning')
        return predict_plist


# In[8]:


def result(df_id, column_name, song_len, tag_len, predict):
        #train_df = df
        df_id = list(df_id)
        col= column_name
        ori_song = col[:song_len]
        ori_tag = col[song_len:]
        #onehot = df_one
        #predict = deep_learning(train)
        print('ok')
        song_predict = predict[:,:song_len]
        tag_predict = predict[:,song_len:]
        print('ok')
        result=[]
        n=0
        for i in df_id:
            dic={}
            dic['id']=i

            plist_song=song_predict[n].argsort()[-100:]
            p_song=[]
            for song in plist_song:
                p_song.append(ori_song[song])
            dic['songs']=p_song

            plist_tag=tag_predict[n].argsort()[-10:]
            p_tag=[]
            for tag in plist_tag:
                p_tag.append(ori_tag[tag])
            dic['tags']=p_tag
            n+=1
            result.append(dic)
        print('result')
        return result


# In[9]:


def write_json(data, fname):
    def _conv(o):
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        raise TypeError

    parent = os.path.dirname(fname+".json")
    distutils.dir_util.mkpath( parent)
    with io.open( fname+".json", "w", encoding="utf-8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)


# In[10]:


train_df=load_train_json('train')
t_col,t_song, t_tag=most_popular(train_df,13)
print(len(t_col))


# In[11]:


train_one=create_onehot(train_df, t_col)


# In[12]:


val_df=load_val_json('test')
val_one=create_onehot(val_df, t_col)


# In[13]:


predict = deep_learing(t_col,train_one, val_one)


# In[14]:


result=result(val_df['id'],t_col, t_song, t_tag,predict )


# In[15]:


write_json(result,'results')


# In[ ]:


'''if __name__ == "__main__":
train_df=load_train_json('C:/Users/dladu/arena_data/orig/val')
t_col,t_song, t_tag=most_popular(train_df,1)
#train_zero =create_zero(t_col)
train_one=create_onehot(train_df, t_col)
    
    val_df=load_train_json('val')
    val_one=create_onehot(val_df, t_col)
    predict = deep_learing(t_col,train_one, val_one)
    result=result(val_df['id'],t_col, t_song, t_tag,predict )
    write_json(result,'results')'''


# In[ ]:




