#%%
#read user reviews
import json, ast
import pandas as pd
from pandas import array
import numpy as np
from pandas.core.series import Series
import os
from sklearn.utils import resample
path = r"D:\Pycharm Cloud\steam data"

os.chdir(path)
# %%
#read steam_reviews
'''
reviews_raw = []
with open('steam_new.json',encoding= 'UTF-8') as f:
    x=1
    for line in f:
        jdata = ast.literal_eval(json.dumps(line)) # Removing uni-code chars
        b=jdata[:-1].replace("u'","'")
        res = ast.literal_eval(b)
        reviews_raw.append(res)
        x+=1
        print(x)
'''
#%%
# read reviews and get random sample

reviews_raw = pd.read_csv('new_reviews.csv')
#reviews_sample = resample(reviews, n_samples=1000000, random_state = 42)

# %%
#read steam_games

games_raw = []
with open('steam_games.json',encoding= 'UTF-8') as f:
    x=1
    for line in f:
        jdata = ast.literal_eval(json.dumps(line)) # Removing uni-code chars
        b=jdata[:-1].replace("u'","'")
        res = ast.literal_eval(b)
        games_raw.append(res)
        x+=1
        print(x)
# %%
# convert to dataframe
reviews = reviews_raw.copy()
games = pd.DataFrame(games_raw)    
# %%
def binary_search(arr, target):
    
    # The starting and ending point of the input array
    left, right = 0, len(arr) - 1
    
    # While the subarray is not empty
    while (left <= right):
        # Get the index of the middle item in the subarray
        mid = left + (right - left) // 2
        
        # If the middle item equals the target
        if arr[mid] == target:
            return mid
        # If the middle item is larger than the target
        elif arr[mid] > target:
            right = mid - 1
        # If the middle item is smaller than the target
        else:
            left = mid + 1
            
    return "Nope"

# %%
# sort game id
games.dropna(subset=['id'],inplace=True)
games['id'] = games['id'].astype(int)
games = games.sort_values('id')
games = games.reset_index()
# change id to int
#reviews = reviews.reset_index()
#reviews['product_id'] = reviews['product_id'].astype(int)
# %%
# add two new cols

#reviews['genres'] = np.nan
#reviews['game'] = np.nan
# %%
# Match
x=1
for i in range(700000,len(reviews)): 
    idx = binary_search(array(games['id']),reviews.iloc[i]['product_id'])
    if idx != 'Nope':
        reviews.loc[i, 'genres'] = str(games.loc[idx,'genres'])
        reviews.loc[i, 'game'] = games.loc[idx,'app_name']
        x+=1
        print(x)


#%%
reviews.to_csv(path + "\\new_reviews.csv")



# %%
len(reviews[reviews['game'].isnull()])

# %%
reviews = reviews.drop(columns=['Unnamed: 0','Unnamed: 0.1', 'index', 'Unnamed: 0.1.1'])
# %%
len(reviews)

# %%
games = pd.DataFrame(games_raw) 
df = games.dropna(subset=['genres'])

#%%
#genre column to list
import ast
genres_l=df['genres'].dropna()

#%%
#multi label encoder
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
mlb.fit_transform(genres_l)   

#check sum of game genres
genre_multi=list(mlb.fit_transform(genres_l))
res = sum(genre_multi, 0)
genre_dict=list(mlb.classes_)
# %%
#print genre numbers
genre_sum={'Type':mlb.classes_,'Num':res}
sum_view=pd.DataFrame(genre_sum).sort_values(by=['Num'],ascending=False).reset_index(drop = True)

print(sum_view)

#%%
a = sum_view['Num'].copy()

#%%
import matplotlib.pyplot as plt
# plot sum
sum_view[["Type","Num"]].plot.bar(x="Type")
# pie plot
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
plt.figure(figsize=(8,8))
grp = sum_view[["Type","Num"]]
plt.pie(grp['Num'], labels=grp['Type'],  startangle=0,
        autopct='%1.1f%%',pctdistance = 0.7,labeldistance=1.2, wedgeprops={'edgecolor':'black'}, colors=colors)
plt.title('Dataset Genres Pie Plot')
plt.savefig('Dataset Genres Pie Plot.png', dpi=300)
plt.show()


# %%
for i in df.genres[0]:
    print(i)
# %%
df.head()
# %%
a
# %%
sum_view = pd.DataFrame(np.transpose(np.vstack((b, a))))
# %%
sum_view['Type']
# %%
b = sum_view['Type'].copy()
# %%
a = a[:-1]

# %%
a
# %%

sum_view = sum_view.rename(columns={0: 'Type', 1:'Num'})
# %%


# %%
