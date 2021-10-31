#%%
import numpy as np
import pandas as pd
from scipy import stats

#steam=pd.read_csv("/Users/chenzichu/Desktop/Capstone/data/df_processed.csv")
steam=pd.read_csv(r"D:\Pycharm Cloud\steam data\df_processed.csv")
#https://medium.com/@osas.usen/topic-extraction-from-tweets-using-lda-a997e4eb0985
#%%
steam = steam.drop(columns=['Unnamed: 0','Unnamed: 0.1'])
steam = steam.loc[steam['length'] >= 10]
steam = steam[(np.abs(stats.zscore(steam['length'])) < 2.5)].reset_index(drop=True)

#%%
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

number_of_topics = 1
lda = LatentDirichletAllocation(n_components=number_of_topics, random_state=45) # random state for reproducibility
vectorizer = TfidfVectorizer()

genre = ['Action', 'Adventure','Casual','Education','Indie',
       'RPG', 'Racing', 'Simulation','Sports','Strategy']
#%%
# handle the 'string list', remove '[ ]'
import re
def replce_s(text):
    text = re.sub(r'[\[\]\,\']','' ,text)
    return text

#%%
print(replce_s("['build', 'tower', 'sound', 'dull', 'varried', 'level', 'sort', 'mechanic', 'good', 'job', 'explaning', 'highly', 'recomend', 'gameplay', 'art', 'music', 'perfect', 'storyline', 'understood', 'bit', 'silly', 'bizare']"))
#%%

lda_stopwords = ['like', 'good', 'fun', 'great', 'really', 'make', 'story', 'buy',
       'love', 'hour', 'recommend', 'lot', 'want', 'gameplay', 'thing',
       'look', 'best', 'say', 'feel', 'character', 'worth', 'think',
       'try', 'way', 'graphic', 'new', 'people', 'level', 'enjoy', 'bad', 'pretty', 
       'work', 'need', 'player', 'use', 'little', 'start', 'bit',  'know', 'come', 
       'nice', 'end', 'hard', 'amazing', 'money', 'far', 'awesome', 'free', 
       'easy', 'review', 'long', 'different', 'kill', 'steam', 'friend', 'run', 
       'puzzle', 'experience', 'control', 'short', 'year', 'price', 'music', 'bug', 'old', 'update',
       'overall', 'add', 'point', 'sale', 'quite', 'actually', 'right',
       'map', 'day', 'wait', 'mode', 'fix', 'dont', 
       'pay','world','combat', 'weapon', 'mechanic', 'multiplayer', 'enemy', 'mod',
       'definitely', 'problem', 'fan', 'simple', 'style',
       'pc', 'base', 'spend', 'change', 'issue', 'server',
       'die', 'life', 'content', 'finish', 'sure', 'fps', 'big', 'cool',
       'highly', 'early', 'turn','challenge', 'interesting', 'real', 'release', 'version',
       'complete', 'probably',  'minute',
        'mission', 'crash', 'expect', 'series',
       'developer', 'stuff', 'original',  'high', 'community',
       'online', 'dlc', 'pick', 'kind', 'mean', 'sound', 'let','build', 'art', 'design', 
       'fight', 'beautiful', 'action', 'maybe',
       'rpg', 'soundtrack', 'beat', 'unique', 'gun', 'tell', 'help',
       'learn', 'fantastic', 'single', 'shooter', 'leave',
       'hope', 'especially', 'amaze', 'team', 'reason', 'access', 'item',
       'lose', 'zombie', 'support', 'alot''adventure','type','wish','stop', 'open', 'devs', 'main', 
       'im', 'away', 'super', 'skill']
#%%
# remove lda stopwords
LDA_Reviews = []
lda_review = []
for i in range(len(steam['Normalized Review'])):
    word_list = []
    for w in ast.literal_eval(steam['Normalized Review'][i]):
        if w not in lda_stopwords:
            word_list.append(w)
        
    LDA_Reviews.append(word_list)
    

#%%
steam['LDA_Review'] = LDA_Reviews
#%%
[' '.join(r) for r in steam[steam['Casual'] == 1]['LDA_Review'] ]
#steam['Normalized Review'][0]
#%%
# LDA fit all genres 
Topic = []

for i in genre:
    row_list = [' '.join(r) for r in steam[steam[i] == 1]['LDA_Review'] ]
    TFIDF = vectorizer.fit_transform(row_list)
    print(TFIDF.shape)

    terms = vectorizer.get_feature_names()
    lda.fit(TFIDF)

    topic  = []
    for comp in lda.components_:
        termsInComp = zip(terms,comp)
        sortedterms = sorted(termsInComp, key=lambda x: x[1],reverse=True)[:30]        
        for term in sortedterms:
            topic.append(term[0])

    Topic.append(topic)

#%%
pd.DataFrame(np.array(Topic).transpose(),columns=genre)

#%%       
np.array(Topic)[0]

# %%
'build', 'art', 'design', 'fight', 'beautiful', 'action', 'maybe',
       'rpg', 'soundtrack', 'beat', 'unique', 'gun', 'tell', 'help',
       'learn', 'fantastic', 'single', 'shooter', 'leave',
       'hope', 'especially', 'amaze', 'team', 'reason', 'access', 'item',
       'lose', 'zombie', 'support', 'alot'