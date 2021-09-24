#%%
#load data
import pandas as pd
df=pd.read_csv('/Users/chenzichu/Desktop/Capstone/steam data/new_reviews.csv')
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
