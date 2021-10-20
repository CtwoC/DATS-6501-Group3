#%%
import numpy as np
import pandas as pd
from scipy import stats

#steam=pd.read_csv("/Users/chenzichu/Desktop/Capstone/data/df_processed.csv")
steam=pd.read_csv(r"D:\Pycharm Cloud\steam data\df_processed.csv")
#https://medium.com/@osas.usen/topic-extraction-from-tweets-using-lda-a997e4eb0985
#%%
steam = steam.drop(columns=['Unnamed: 0','Unnamed: 0.1'])
steam = steam.loc[steam['length'] >= 10].reset_index(drop=(True))
steam = steam[(np.abs(stats.zscore(steam['length'])) < 2.5)]
#%%
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

number_of_topics = 1
lda = LatentDirichletAllocation(n_components=number_of_topics, random_state=45) # random state for reproducibility


genre = ['Action', 'Adventure','Casual','Education','Indie',
       'RPG', 'Racing', 'Simulation','Sports','Strategy']
#%%
Topic = []

for i in genre:
    word_list = [w for r in steam[steam[i] == 1]['Normalized Review'] for w in ast.literal_eval(r)]
    corpus = word_list
    vectorizer = TfidfVectorizer()
    TFIDF = vectorizer.fit_transform(corpus)
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
pd.DataFrame(Topic,columns=genre)

#%%       

#%%


# %%

number_of_topics = 1
model = LatentDirichletAllocation(n_components=number_of_topics, random_state=45) # random state for reproducibility
# Fit data to model
model.fit(tf)
# %%
# Log Likelyhood: Higher the better
print("Log Likelihood: ", model.score(tf))

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", model.perplexity(tf))

# See model parameters
print(model.get_params())
# %%
import pickle
# pickle baseline model
filename = 'models/baseline_lda_model.sav'
pickle.dump(model, open(filename, 'wb'))
# %%
#finetuning

#%%
best_lda_model=model
topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]
# Topic-Keyword Matrix
df_topic_keywords = pd.DataFrame(best_lda_model.components_)

# Assign Column and Index
df_topic_keywords.columns = vectorizer.get_feature_names()
df_topic_keywords.index = topicnames

# View
df_topic_keywords.head()
# %%
# Show top n keywords for each topic
def show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=15)        

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords
# %%
# save the model to disk
filename = 'lda_model.sav'
pickle.dump(model, open(filename, 'wb'))
 

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

# %%
