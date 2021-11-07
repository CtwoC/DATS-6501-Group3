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
# Create stopwords

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words
nltk_stop_words = nltk.corpus.stopwords.words('english')

stop_words = set(nltk_stop_words).union(sklearn_stop_words)
print(len(stop_words))

# add own stopwords
stop_words.update(['game','play','time', 'player'])
print(len(stop_words))

#%%
from nltk.corpus import wordnet

# pos_tag convert
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
import re
def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)
# %%
# Tokenize
import nltk
from nltk import TweetTokenizer
from textblob import TextBlob

Tt_Tokenizer = TweetTokenizer()
wnl = nltk.WordNetLemmatizer()

""" select spell checkers & load """
checker = BertChecker()
checker.from_pretrained()
# %%

# set minimum words number
length_lim = 0

x=0
normal_reviews = []
length =[]

for i in range(len(df)):

    txt  = remove_emoji(df.iloc[i]['text'])
    # correct missspelling
    checker.correct(txt)
    # tokenize
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

# %%


df.to_csv('df_processed_correct.csv')

# %%



# %%

# %%
