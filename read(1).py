#%%
#read user reviews
import json, ast
import pandas as pd
from pandas import array
import numpy as np
from pandas.core.series import Series
import os
path = r"G:\新建文件夹"

os.chdir(path)
# %%
#read steam_reviews

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
reviews = pd.DataFrame(reviews_raw)
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
reviews['product_id'] = reviews['product_id'].astype(int)
# %%
# add two new cols

reviews['genres'] = np.nan
reviews['game'] = np.nan
# %%
for i in range(len(reviews)):
    idx = binary_search(array(games['id']),reviews.iloc[i]['product_id'])
    if idx != 'Nope':
        reviews.loc[i, 'genres'] = str(games.loc[idx,'genres'])
        reviews.loc[i, 'game'] = games.loc[idx,'app_name']

#%%
reviews.to_csv(path + "\new_reviews.csv")

# %%
