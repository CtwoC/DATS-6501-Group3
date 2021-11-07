#%%
import numpy as np
import pandas as pd
from scipy import stats
import nltk
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

number_of_topics = 5
lda = LatentDirichletAllocation(n_components=number_of_topics, random_state=45) # random state for reproducibility
vectorizer = TfidfVectorizer()
words = set(nltk.corpus.words.words())

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
       'im', 'away', 'super', 'skill', 'Action', 'Adventure','Casual','Education','Indie',
       'RPG', 'Racing', 'Simulation','Sports','Strategy','genre']
#%%
# remove lda stopwords
LDA_Reviews = []
lda_review = []
for i in range(len(steam['Normalized Review'])):
    word_list = []
    for w in ast.literal_eval(steam['Normalized Review'][i]):
        if w not in lda_stopwords and w in words and w.isalpha():
            word_list.append(w)
        
    LDA_Reviews.append(word_list)
    

#%%
steam['LDA_Review'] = LDA_Reviews
#%%
testdf = [' '.join(r) for r in steam[steam['Casual'] == 1]['LDA_Review'] ]
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
lda_result = pd.DataFrame(np.array(Topic).transpose(),columns=genre)
#%%
lda_result
#%%       
np.array(Topic)[0]

# %%
'build', 'art', 'design', 'fight', 'beautiful', 'action', 'maybe',
       'rpg', 'soundtrack', 'beat', 'unique', 'gun', 'tell', 'help',
       'learn', 'fantastic', 'single', 'shooter', 'leave',
       'hope', 'especially', 'amaze', 'team', 'reason', 'access', 'item',
       'lose', 'zombie', 'support', 'alot'


#%%
lda_result.to_csv('lda_result33.csv')
# %%
import pyLDAvis
pyLDAvis.enable_notebook()
import lda2vec
#%%
# show lda2vec result
npz = np.load(open('topics.pyldavis.npz', 'r'))
dat = {k: v for (k, v) in npz.iteritems()}
dat['vocab'] = dat['vocab'].tolist()
# dat['term_frequency'] = dat['term_frequency'] * 1.0 / dat['term_frequency'].sum()
#%%
top_n = 10
topic_to_topwords = {}
for j, topic_to_word in enumerate(dat['topic_term_dists']):
    top = np.argsort(topic_to_word)[::-1][:top_n]
    msg = 'Topic %i '  % j
    top_words = [dat['vocab'][i].strip()[:35] for i in top]
    msg += ' '.join(top_words)
    print (msg)
    topic_to_topwords[j] = top_words
