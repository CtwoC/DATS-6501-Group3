#%%
#load data
import pandas as pd
df=pd.read_csv(r"D:\Pycharm Cloud\\steam data\new_reviews.csv")
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
sum_view=pd.DataFrame(genre_sum).sort_values(by=['Num'],ascending=False).reset_index(drop = True)
print(sum_view)
#%%
#sum_view['Num'] = a
#sum_view
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
plt.title('Sample Genres Pie Plot')
plt.savefig('Sample Genres Pie Plot.png', dpi=300)
plt.show()
#%%
#genres lists to df, this takes a few minutes
import numpy as np
genre_df=pd.DataFrame(np.vstack(genre_multi))
genre_df.columns=mlb.classes_
# %%
genre_corr = genre_df.drop(columns=['Animation &amp; Modeling','Audio Production','Design &amp; Illustration','Photo Editing','Software Training',
'Utilities','Video Production','Web Publishing','Education'])
#%%
#visualize correlation

correlation=genre_corr.corr()
correlation.style.background_gradient(cmap='coolwarm')

#%%
import seaborn as sns
fig, ax = plt.subplots()
plt.figure(figsize=(16,6))
sns.heatmap(genre_corr.corr(), fmt='.4f', 
            cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
plt.savefig('result.png', pad_inches=0.0)

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
new_df.head()
# %%
# save to local
df.to_csv('new_df.csv', index=False)

# %%
# Genre Casual

Casual = new_df[new_df['Casual'] == 1]

#%%

# Cleaning

def review_clean(df):

    # 直接删除评论列中的空值（不包含空字符串）
    df = df.dropna(subset=['text'])

    # 根据id与text两列作为参照，如存在用户id与text同时相同，那么只保留最开始出现的。
    df.drop_duplicates(subset=['text'], keep='first', inplace=True)
    # 重置索引
    df.reset_index(drop=True, inplace=True)

    # 用空字符串('')替换纯数字('123')
    df['text'] = df['text'].str.replace('^[0-9]*$', '')

    # 将开头连续重复的部分替换为空''
    prefix_series = df['text'].str.replace(r'(.)\1+$', '')
    # 将结尾连续重复的部分替换为空''
    suffix_series = df['text'].str.replace(r'^(.)\1+', '')

    for index in range(len(df['text'])):
        # 对开头连续重复的只保留重复内容的一个字符(如'aaabdc'->'abdc')
        if prefix_series[index] != df['text'][index]:
            char = df['text'][index][-1]
            df['text'][index] = prefix_series[index] + char
        # 对结尾连续重复的只保留重复内容的一个字符(如'bdcaaa'->'bdca')
        elif suffix_series[index] != df['text'][index]:
            char = df['text'][index][0]
            df['text'][index] = char + suffix_series[index]

    # 将空字符串转为'np.nan',即NAN,用于下一步删除这些评论
    df['text'].replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)
    # 删除comment中的空值，并重置索引
    df = df.dropna(subset=['text'])
    df.reset_index(drop=True, inplace=True)

# %%
review_clean(Casual)

Casual = Casual.dropna(subset=['text'])
# %%
# Normalization

import nltk
from nltk import word_tokenize

normal_review = []
# wnl = nltk.WordNetLemmatizer()

for i in range(len(Casual)):
    tokens = word_tokenize(Casual.iloc[i]['text'])
    
    normal_review.append(' '.join([t.lower() for t in tokens if t.isalpha()]))

Casual['Normalized Review'] = normal_review

# %%
Casual.to_csv('Casual_df.csv', index=False)
# %%
#check if all value in text column are string
all(isinstance(item, str) for item in casual['Normalizaed Review'])

def checker(txt):
    try:
        float(txt)
        return False
    except:
        return True

casual=casual[casual['Normalizaed Review'].apply(checker)]