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
sub=['text']+list(new_df.columns[15:])
new_df=new_df[sub]

#drop rows include none of these genres
#sub2=["Action","Indie","Adventure","RPG","Strategy","Simulation","Casual","Sports","Racing"]
#new_df=new_df.loc[(new_df[sub2] != 0).any(axis=1)]


#%%
import os
os.getcwd()
os.chdir(r"D:\Pycharm Cloud\steam data")
new_df.to_csv('new_df.csv')







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

df.head()
#%%
# get subset for test

df = df[df['df'] == 1]

df.head()

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
# %%
# Tokenize
import nltk
from nltk import TweetTokenizer
from textblob import TextBlob

Tt_Tokenizer = TweetTokenizer()
stopwords = nltk.corpus.stopwords.words('english')
wnl = nltk.WordNetLemmatizer()
#%%
from nltk.corpus import wordnet

# pos convert
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

#%%
# set minimum words number
length_lim = 0

x=0
normal_reviews = []
length =[]

for i in range(len(df)):

    txt  = df.iloc[i]['text']
    # correct missspelling
    #txt = TextBlob(txt)
    #textCorrected = txt.correct()

    tokens = Tt_Tokenizer.tokenize(txt)

    normal_review = []
    # lemma
    # lower
    # stopwords
    for t in nltk.pos_tag(tokens):
        pos_tag = get_wordnet_pos(t[1])    
        if (t[0].isalpha()) and (pos_tag != None):
            lemma_word = wnl.lemmatize(t[0].lower(), pos=pos_tag)
            if lemma_word not in stop_words: 
                normal_review.append(lemma_word)

        if (t[0].isalpha()) and (pos_tag == None) and (t[0].lower() not in stop_words):
            normal_review.append(t[0].lower())

    # length
    if len(normal_review) >= length_lim:
        normal_reviews.append(normal_review)

    length.append(len(normal_review))

    x += 1
    print(x)
#%%
df['Normalized Review'] = normal_reviews
df['length'] = length
df = df[df['length'] != 0]
#%%
all(isinstance(item, str) for item in df['text'])
# %%
# Length analysis

df['length'].hist(bins=100,range=[1,300])

print(df['length'].describe())

print(len(df[df['length']==2]))
#%%
# Length remove
df
# Based on histgram(distribution)



# Based on outlier

# only remove large length

# outlier_removed = df[(np.abs(stats.zscore(df['length'])) < 2.5)]
#%%
# df.to_csv('df_df.csv', index=False)

# %%
len(df)
# %%
df.head()
# %%

# %%

# %%


df.to_csv('df_processed.csv')
# %%
stop_words
# %%
