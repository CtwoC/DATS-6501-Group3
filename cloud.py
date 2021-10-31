#%%
import pandas as pd
from wordcloud import WordCloud

casual=pd.read_csv('/Users/chenzichu/Desktop/Capstone/data/Casual_df.csv')
all(isinstance(item, str) for item in casual['Normalizaed Review'])

def checker(txt):
    try:
        float(txt)
        return False
    except:
        return True

casual=casual[casual['Normalizaed Review'].apply(checker)]
text_list=list(casual["Normalizaed Review"])
text=""
for t in text_list:
    text+=t
    text+=" "

wordcloud = WordCloud(min_font_size=10, width = 800, height = 400).generate(text)

import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
# %%
