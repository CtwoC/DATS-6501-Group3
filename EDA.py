#%%
#load data
import pandas as pd
#df=pd.read_csv("/Users/chenzichu/Desktop/Capstone/data/new_reviews.csv")
df=pd.read_csv(r"D:\Pycharm Cloud\steam data\new_reviews.csv")
#view
df.head()
df.describe()
df['genres'].value_counts(dropna=True)
#%%
#genre column to list
import ast
genres_s=df['genres'].dropna().tolist()
genres_l=[]
for genres in genres_s:
    try:
        genres_l.append(ast.literal_eval(genres))
    except:
        print(genres)

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
sum_view=pd.DataFrame(genre_sum)
print(sum_view.sort_values(by=['Num'],ascending=False))
#plot sum
sum_view[["Type","Num"]].plot.bar(x="Type")
#%%
#genres lists to df, this takes a few minutes
import numpy as np
genre_df=pd.DataFrame(np.vstack(genre_multi))
genre_df.columns=mlb.classes_
# %%
#visualize correlation
correlation=genre_df.corr()
correlation.style.background_gradient(cmap='coolwarm')
# %%
#concat genre encoded columns 
df=df.dropna(subset=["genres"])
new_df=df.join(genre_df)
#choose columns
sub=["text","Action","Indie","Adventure","RPG","Strategy","Simulation","Casual","Sports","Racing"]
new_df=new_df[sub]
#drop rows include none of these genres
sub2=["Action","Indie","Adventure","RPG","Strategy","Simulation","Casual","Sports","Racing"]
new_df=new_df.loc[(new_df[sub2] != 0).any(axis=1)]


#%%
new_df.to_csv('new_df.csv')

#%%
import os
os.getcwd()





# %%
# load new df(splited label)
import pandas as pd
import matplotlib as plt
import numpy as np
from scipy import stats

df=pd.read_csv(r"D:\Pycharm Cloud\steam data\new_df.csv")
df.info()
#%%
# train test split


#%%

# Pandas Basic Cleaning

def review_clean(df):

    # 直接删除评论列中的空值（不包含空字符串）
    df.dropna(inplace=True)

    # 根据id与text两列作为参照，如存在用户id与text同时相同，那么只保留最开始出现的。
    df.drop_duplicates(subset=['text'], keep='first', inplace=True)
    # 重置索引
    df.reset_index(drop=True, inplace=True)

    # 用空字符串('')替换纯数字('123')
    df['text'] = df['text'].str.replace('^[0-9]*$', '', regex =True)

    # 将空字符串转为'np.nan',即NAN,用于下一步删除这些评论
    df['text'].replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)
    
    # 删除空值，并重置索引
    df.dropna(subset=['text'], inplace=True)
    df.reset_index(drop=True, inplace=True)

# %%
review_clean(df)

#%%
# add length column(takes a few minutes)

length =[]
for i in range(len(df)):
    length.append(len(df.iloc[i]['text']))
df['length'] = length


# %%
# Length analysis



df['length'].hist(bins=1000, range=[0, 50])
#%%
# Length remove

# Based on histgram(distribution)



# Based on outlier

# only remove large length

# outlier_removed = df[(np.abs(stats.zscore(df['length'])) < 2.5)]
#%%
# get subset for test

Casual = df[df['Casual'] == 1]

Casual.head()

#%%
# Create stopwords

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words
nltk_stop_words = nltk.corpus.stopwords.words('english')

stop_words = set(nltk_stop_words).union(sklearn_stop_words)
print(len(stop_words))

# add own stopwords
stop_words.update(['game','play','time'])
print(len(stop_words))
#%%
print(stop_words)

# %%
# Tokenize

# lemma
# lower
# alpha
# stopwords

from textblob import TextBlob
textCorrected = textBlb.correct()


import nltk
from nltk import TweetTokenizer

Tt_Tokenizer = TweetTokenizer()
stopwords = nltk.corpus.stopwords.words('english')
wnl = nltk.WordNetLemmatizer()

normal_review = []

for i in range(len(Casual)):
    tokens = Tt_Tokenizer(Casual.iloc[i]['text'])
    for t in tokens:
        if (t.isalpha()) and (t.lower() not in stopwords) and

    
    normal_review.append(' '.join([wnl.lemmatize(t.lower()) for t in tokens if t.isalpha()]))

Casual['Normalized Review'] = normal_review


#%%
all(isinstance(item, str) for item in Casual['text'])
# %%
#%%
# Casual.to_csv('Casual_df.csv', index=False)