# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:03:37 2019

@author: User
"""


import numpy as np 
import pandas as pd 
from sklearn.externals import joblib   
from sklearn.preprocessing import scale
df_clusing=pd.read_csv('total_cluster.csv')
df_clusing['cluster_label']=df_clusing.user_label.map(str)+'|'+df_clusing.song_label.map(str)
cluster_label=pd.DataFrame(df_clusing.cluster_label.value_counts())
dir_cluster_label=cluster_label.to_dict()
df_clusing['rating']=df_clusing.cluster_label.map( dir_cluster_label['cluster_label'])

#測試song_id是不是一樣
        
rating_value=df_clusing.rating.value_counts()
df_clusing['rating_test']=scale(df_clusing.rating)
df_clusing['rating_test']=np.round(df_clusing.rating_test)
df_clusing.rating_test.value_counts()


songinfo=pd.read_csv('songSearch.csv')
songinfo.rename(columns={songinfo.columns[1]:'iid'},inplace=True)
songinfo=songinfo[['iid','name']]
 #3.56

#model.predict(uid,iid)


#得到前20手推薦歌單
df=df_clusing['song_id'].unique()
df2=df_clusing['user_id'].unique()
from surprise import SVD
model01=SVD(n_factors=100)
model01=joblib.load('recommend.pkl')
rlist=[]
from tkinter import *
import tkinter.messagebox 

def user_get_recommend():
    tkinter.messagebox.showinfo(title='Please! wait for minutes', message='請稍後！')

    print('test')
    global rlist,rlist1,num01,model01
    x=num01.get()
    x=int(x)
    for i in range(len(df)):
        a=model01.predict(uid=x,iid=i)
        rlist.append(a)
    rlist1=pd.DataFrame(rlist).sort_values(by='est',ascending=False).head(20)
    rlist=[]
    print('test1')

    rlist=rlist1[['iid']]
    rlist=pd.merge(rlist,songinfo,on='iid')
    print('test2')

    rcm_result=rlist['name']
    rcm_result=rcm_result.tolist()
    print('test3')

    rcm_result=str(rcm_result)
    print(rcm_result)
    result01.set(rcm_result)
    rlist=[]
    
def song_get_recommend():
    tkinter.messagebox.showinfo(title='Please! wait for minutes', message='請稍後！')

    print('test')
    global rlist,rlist1,num02,model01
    x=num02.get()
    x=int(x)
    for i in range(len(df2)):
        a=model01.predict(uid=i,iid=x)
        rlist.append(a)
    rlist1=pd.DataFrame(rlist).sort_values(by='est',ascending=False).head(20)
    rlist=[]
    print('test1')

    rlist=rlist1[['iid']]

    result02.set(rlist)
    rlist=[]    
    

window01=Tk()
window01.geometry("1200x400")
window01.title('菜逼八 recommend system')
num01=IntVar()
result01=StringVar()
Label(window01,width=10,text="input user_id:").grid(row=0,column=0)
Entry(window01,width=10,textvariable=num01).grid(row=0,column=1)
Button(window01,width=20,text="start recommend",command=user_get_recommend,bg='orange', fg='red').grid(row=2,column=0)
Label(window01,width=50,height=15,wraplength =200,textvariable=result01,bg='khaki1').grid(row=3,column=1)

num02=IntVar()
result02=StringVar()
Label(window01,width=10,text="input song_id:").grid(row=0,column=2)
Entry(window01,width=10,textvariable=num02).grid(row=0,column=3)
Button(window01,width=20,text="start recommend",command=song_get_recommend,bg='orange', fg='red').grid(row=2,column=2)
Label(window01,width=50,height=15,wraplength =200,textvariable=result02,bg='khaki1').grid(row=3,column=3)

window01.mainloop() 
    
    

#檢視 trainset



#得到前20手推薦歌單

#===========================================================
#step 3. Evaluation
#surprise.accuracy ,RMSE
from surprise import accuracy
accuracy.rmae(predictions) #RMSE: 0.8936

#===========================================================
#step 4. 使用模型做推薦
#model.estimate(u,i), 預測u=uid, i=iid 的推薦 rating




