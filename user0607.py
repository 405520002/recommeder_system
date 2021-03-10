# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 01:09:58 2019

@author: user
"""

 
import numpy as np 
import pandas as pd 
from sklearn import svm,tree
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
df=pd.read_csv('k2.csv') 
df = df.dropna(axis = 1) 
df_msno=df[['msno']] 
df_msno=df_msno.drop_duplicates('msno') 
df_msno.reset_index(inplace=True) 
df_msno.reset_index(inplace=True) 
dfmsno_t=df_msno.tail(5) 
df_msno.rename(columns={df_msno.columns[0]:'user_id'},inplace=True) 
df_msno=df_msno.drop(columns='index')

df_song=df[['song_id']] 
df_song=df_song.drop_duplicates('song_id') 
df_song.reset_index(inplace=True) 
df_song.reset_index(inplace=True)
df_song.rename(columns={df_song.columns[0]:'song_number'},inplace=True) 
dfst=df_song.tail(5)
df_song=df_song.drop(columns='index')

df1=pd.merge(df,df_song,on='song_id') 
df1=pd.merge(df1,df_msno,on='msno') 
df1=df1.drop(columns=['Unnamed: 0', 'level_0', 'msno', 'song_id']) 
df1=df1.drop(columns=['lyricist','registration_init_time'])#drop不是整數的欄位 
df1=df1.drop(columns=['song_id_y','artist_name']) 
df1=df1.drop(columns=['lyricst_index','song_length']) 
df2=df1[df1['target']==1]
 

print(len(df2.user_id.unique())) #看user的量
user_count=df2.user_id.value_counts()
user_array=np.array(user_count[:2000].index)
user_df=pd.DataFrame(user_array)
user_df.rename(columns={user_df.columns[0]:'user_id'},inplace=True)
df_user=pd.merge(user_df,df2,on='user_id')



                 #)==========================================
df_user=df_user.drop(columns='target')
df_train_x=df_user.drop(columns=['user_id','song_number'])
df_train_y=df_user['user_id']
user_array=np.array(user_count[2000:].index)
df_test_y=pd.DataFrame(user_array)
df_test_y.rename(columns={df_test_y.columns[0]:'user_id'},inplace=True)
df_test_user=pd.merge(df_test_y,df2,on='user_id')
df_rest=df_test_user
dft=df_test_user.head(5)
df_test_user=df_test_user.drop(columns='target')
df_test_user=df_test_user.drop(columns=['user_id','song_number'])





from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn=knn.fit(df_train_x, df_train_y)
pre_user_id=knn.predict(df_test_user)
pre_user_id=pd.DataFrame(pre_user_id)
pre_user_id.rename(columns={pre_user_id.columns[0]:'user_id'},inplace=True)
dfu=df_user[['user_id']]

df_add=pd.concat([dfu,pre_user_id],axis=0)
df_label=df_add.drop_duplicates()
df_label.reset_index(inplace=True)
df_label.reset_index(inplace=True)
df_label=df_label.drop(columns='index')
df_label.rename(columns={df_label.columns[0]:'user_label'},inplace=True)
df_add=pd.merge(df_add,df_label,on='user_id')
df_add=df_add.drop(columns='user_id')
df=pd.concat([df_user,df_rest],axis=0)
df.reset_index(inplace=True)
df.reset_index(inplace=True)
df_add.reset_index(inplace=True)
df_add.reset_index(inplace=True)
df_add=df_add.drop(columns='index')
df=df.drop(columns='index')
df_final=pd.merge(df_add,df,on='level_0')
df_final=df_final[['user_id','user_label','song_number','target']]
#df_final.to_csv('user_cluster.csv')

joblib.dump(knn, 'user.pkl')
