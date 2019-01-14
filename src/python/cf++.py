
# coding: utf-8
# # データ前処理
import pandas as pd
import numpy as np
import copy 
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

logs = pd.read_csv('../sample/medium/log_medium.csv')
users = pd.read_csv('../sample/medium/user_medium.csv')
products = pd.read_csv('../sample/medium/product_medium.csv')

df_u = users.drop(['age','address','last_login','last_purchase'], axis = 'columns')
df_p = products.drop(['shop','tag_price'],axis='columns')

category_list = {
    "tops":0,
    "pants":1,
    "outer":2,
    "bag":3,
    "shoes":4
}
df_p["categoryId"] = df_p["category"].map(category_list)

df_po = pd.merge(df_p,logs, on='product_id')
df_pud = pd.merge(df_po,df_u, on='user_id',how='inner')
df_pu = df_pud.drop(['product_id','sub_category','order_id', 'user_id','price','tag_price'],axis='columns')

#割引カラムとセール商品カラムの追加
df_pud['discount'] = df_pud['price']/df_pud['tag_price']
for df_pud in [df_pud]:
    df_pud['isSaled'] = 0
    df_pud.loc[df_pud['discount'] < 1, 'isSaled'] = 1
df_pud = df_pud.drop(['product_id','sub_category', 'user_id','order_id','price','tag_price','discount'],axis='columns')
df_pud_pivot = df_pud.pivot(index='user_name', columns = 'product_name',values = 'quantity' ).fillna(0)


# ### calc_posDegree_for_sale_product関数を用いユーザーのセールへのポジティブ度を計算する
# セール商品へのポジティブ度を計算する関数
def calc_posDegree_for_sale_product(pro_list,user):
    rownum = len(pro_list[pro_list['user_name'] == user])
    saled_cnt = pro_list[pro_list['user_name'] == user]['isSaled'].sum()
    posDegree = saled_cnt/rownum
    return(posDegree)

# # レコメンド関数

#入力
matrix = df_pud_pivot
matrix_sparse = csr_matrix(df_pud_pivot.values)
product_list = df_pud

# Scikit-learnのライブラリを利用しモデルを作成
N = 10
knn = NearestNeighbors(n_neighbors=N,algorithm= 'brute', metric= 'cosine')
# 前処理したデータセットでモデルを訓練
model_knn = knn.fit(matrix_sparse)

#類似度を求める関数
def get_sim(matrix,user1,user2):
    distance, indice = model_knn.kneighbors(matrix.iloc[matrix.index== user1].values.reshape(1,-1),n_neighbors=N)
    for i in range(0, len(distance.flatten())):
        if  i > 0:
            if matrix.index[indice.flatten()[i]] == user2:
                return(1 - distance.flatten()[i])

#商品集合を求める関数
def get_product_set(user):
    s = set([])
    for i in range(len(matrix.loc[user].values)):
        if matrix.loc[user].values[i] > 0:
            s.add(matrix.columns[i])
    return(s)

#セール商品かどうかのチェック関数
def check_sale(product_name, product_list):
    return(product_list[product_list['product_name'] ==  product_name]['isSaled'].values[0])

def get_recommend(user, top_N,mode):
    totals = {}  ; simSums = {}
    # 全てのユーザー、商品リストの作成
    list_product = []
    list_user = []
    for i in range(len(matrix.values)):
        list_product.append(matrix.columns[i])
        list_user.append(matrix.index[i])
    
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
        sim = get_sim(matrix, user,other)
        if sim is not None:
            for item in set_new_product:
                #類似度 *  
                totals.setdefault(item,0)
                score = matrix.loc[other,item]*sim 
                if mode == 1:
                    if check_sale(item,product_list):
                        posdeg_for_sale= calc_posDegree_for_sale_product(product_list, user)
                        score = score*(1 + posdeg_for_sale)
                totals[item] += score
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

print("Aさんへのレコメンド:",get_recommend('A',5,1))
print("Kさんへのレコメンド:",get_recommend('K',5,1))