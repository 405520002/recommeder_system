# -*- coding: utf-8 -*-
"""
Created on Wed Jun 5 01:09:58 2019

@author: user
"""

from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
import pandas as pd 
from sklearn.externals import joblib

#user
df_userFin=pd.read_csv('song_cluster.csv')
df_user=df_userFin[df_userFin['target']==1]
df_user.rename(columns={'msno':'user_id'},inplace=True)
print(len(df_user.user_id.unique())) #看user的量
user_count=df_user.user_id.value_counts()
user_array=np.array(user_count[:2000].index)
user_df=pd.DataFrame(user_array)
user_df.rename(columns={user_df.columns[0]:'user_id'},inplace=True)
df_user_1=pd.merge(user_df,df_user,on='user_id')

feature=['city', 'bd', 'registered_via',
       'expiration_date', 'source_system_tab', 'source_screen_name',
       'source_type', 'gendertype', 'membership_days', 'registration_year',
       'registration_month', 'registration_date', 'expiration_year',
       'expiration_month',
       'registration_init_time']
knn=KNeighborsClassifier().fit(df_user_1[feature],df_user_1.user_id)

#挑predict的資料
user_array=np.array(user_count[2000:].index)
user_df1=pd.DataFrame(user_array)
user_df1.rename(columns={user_df1.columns[0]:'user_id'},inplace=True)
df_user_2=pd.merge(user_df1,df_user,on='user_id')

pre_user=knn.predict(df_user_2[feature])
df_user_2['user_clust']=pre_user
df_user_1['user_clust']=df_user_1['user_id']
df_user_2=df_user_2.loc[:,['user_id', 'Unnamed: 0', 'song_id', 'Unnamed: 0.1', 'target', 'city',
       'bd', 'registered_via', 'expiration_date', 'source_system_tab',
       'source_screen_name', 'source_type', 'gendertype', 'song_length',
       'language', 'song_year', 'lyricst_index', 'membership_days',
       'registration_year', 'registration_month', 'registration_date',
       'expiration_year', 'expiration_month', 'lyricist_count', 'artist_name',
       'registration_init_time', 'name', 'isrc', 'genre_ids', 'composer',
       'song_clust', 'song_label', 'user_clust']]

df_clusing=pd.concat([df_user_1,df_user_2],axis=0)
df_clusing.sort_values(by=['user_clust'])
#建立轉換字典
def getMap(df,col):
    name = df[col].value_counts().index
    mappping = {}
    for i,j in zip(name,range(len(name))):
            mappping[i]=j
    return mappping

#將colMap欄位轉換成數值
def finMap(df,colNew,colMap,needDict):
    df[colNew]=df[colMap].map(needDict)
    return df
mapp=getMap(df_clusing,'user_clust')
df_clusing=finMap(df_clusing,'user_label','user_clust',mapp)
a=df_clusing.user_clust==df_clusing.user_id
print(a.sum()) #要等於前兩千的個數1965063


df_clusing.to_csv('total_cluster.csv')
joblib.dump(knn, 'user.pkl')
