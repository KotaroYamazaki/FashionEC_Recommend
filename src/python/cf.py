
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


# # データの前処理

# In[2]:

logs = pd.read_csv('../sample/medium/log_medium.csv')
users = pd.read_csv('../sample/medium/user_medium.csv')
products = pd.read_csv('../sample/medium/product_medium.csv')


# In[3]:

logs


# In[4]:

users.tail()


# In[5]:

products.head()


# # データ整形
# 不要なカラムを削る

# In[6]:

df_u = users.drop(['age','address','last_login','last_purchase'], axis = 'columns')
df_p = products.drop(['category','shop','brand','tag_price'],axis='columns')
df_po = pd.merge(df_p,logs, on='product_id')
df_pu = pd.merge(df_po,df_u, on='user_id',how='inner')
df_pu = df_pu.drop(['product_id','sub_category','order_id', 'user_id','price','tag_price'],axis='columns')


# ## ユーザー*商品の行列を作成
# 値は購入した個数

# In[7]:

df_pu_pivot = df_pu.pivot(index='user_name', columns = 'product_name',values = 'quantity').fillna(0)
df_pu_pivot


# In[8]:

df_pu_pivot.loc['A'] > 0


# # 学習
# ## アプローチ
# 1. k近傍法を用いユーザー同士の距離を計算
# 1. 距離から類似度を求め、レコメンド関数を作成
# 1. レコメンド関数にレコメンドしたいユーザーを入力として渡し、商品のレコメンドランキングを返す

# In[9]:

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

#疎行列変換
df_pu_pivot_sparse = csr_matrix(df_pu_pivot.values)


# In[10]:

# Scikit-learnのライブラリを利用しモデルを作成
N = 5
knn = NearestNeighbors(n_neighbors=N,algorithm= 'brute', metric= 'cosine')
 
# 前処理したデータセットでモデルを訓練
model_knn = knn.fit(df_pu_pivot_sparse)


# ### ユーザーの検索関数

# In[11]:

def search_user(string):
    print(df_pu_pivot[df_pu_pivot.index.str.contains(string)].index[0:])


# ### アイテムの検索関数

# In[12]:

def search_item(string):
    print(df_pu_pivot.columns[df_pu_pivot.columns.str.contains(string)])


# #### 検索関数使用例

# In[13]:

search_item('s')
search_user('A')


# ### 類似度を求める関数

# In[14]:

def get_sim(user1,user2):
    distance, indice = model_knn.kneighbors(df_pu_pivot.iloc[df_pu_pivot.index== user1].values.reshape(1,-1),n_neighbors=N)
    for i in range(0, len(distance.flatten())):
        if  i > 0:
            if df_pu_pivot.index[indice.flatten()[i]] == user2:
                return(1 - distance.flatten()[i])


# In[15]:

#使用例　ユーザーBとAの購入ログに基づく類似度
print('simirality(A,G) = ',get_sim('B','C'))


# ### 購入商品を集合にする関数

# In[16]:

def get_product_set(user):
    s = set([])
    for i in range(len(df_pu_pivot.loc[user].values)):
           if df_pu_pivot.loc[user].values[i] > 0:
                s.add(df_pu_pivot.columns[i])
    return(s)


# ### レコメンド関数

# In[17]:

import copy 
def get_recommend(user, top_N):
    totals = {}  ; simSums = {}
    # 全てのユーザー、商品リストの作成
    list_product = []
    list_user = []
    for i in range(len(df_pu_pivot.values)):
        list_product.append(df_pu_pivot.columns[i])
        list_user.append(df_pu_pivot.index[i])
    
    #自分以外のユーザーリスト
    list_others = copy.copy(list_user)
    list_others.remove(user)
    
    # 自分の購入商品集合
    set_user = get_product_set(user)
    
    for other in list_others:
        #本人がまだ購入していない商品の集合を取得
        set_other = get_product_set(other)
        set_new_product = set_other.difference(set_user)
        
        #あるユーザーと本人の類似度を計算
        sim = get_sim(user,other)
        
        if sim is not None:
            for item in set_new_product:
                #類似度 *  
                totals.setdefault(item,0)
                totals[item] += df_pu_pivot.loc[other,item]*sim
                #ユーザーの類似度の積算値
                simSums.setdefault(item,0)
                simSums[item] += sim

    rankings = []
    #ランキングリストの作成
    for item,total in totals.items():
        if simSums[item] != 0:
            rankings.append((total/simSums[item],item))
    rankings.sort()
    rankings.reverse()
    return ([i[1] for i in rankings][:top_N])


# In[18]:

get_recommend('C',6)


# #### 参考
# - Pythonで簡単な協調フィルタリングを実装するためのノート[https://qiita.com/hik0107/items/96c483afd6fb2f077985]
# - 機械学習を使って630万件のレビューに基づいたアニメのレコメンド機能を作ってみよう（機械学習 k近傍法 初心者向け）
# [https://www.codexa.net/collaborative-filtering-k-nearest-neighbor/]
