#https://github.com/NBrisbon/Silmarillion-NLP/blob/master/Sentiment_Analysis.py

import pandas as pd
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm_notebook as tqdm
from tqdm import trange

def text_emotion(df, column):
    '''
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame with ten new columns for each emotion
    '''

    new_df = df.copy()

    xlsx = pd.read_excel('./robinhood/cleaned_data.xlsx')
    emolex_df = xlsx[['word', 'Positive_NRC','Negative_NRC','Anger', 'Anticipation', 'Disgust', 'Fear','Joy',
                      'Sadness', 'Surprise', 'Trust']]
    emotions = emolex_df.columns.drop('word')
    emo_df = pd.DataFrame(0, index=df.index, columns=emotions)

    stemmer = SnowballStemmer("english")

    
    with tqdm(total=len(list(new_df.iterrows()))) as pbar:
        for i, row in new_df.iterrows():
            pbar.update(1)
            document = word_tokenize(new_df.loc[i][column])
            for word in document:
                word = stemmer.stem(word.lower())
                emo_score = emolex_df[emolex_df.word == word]
                if not emo_score.empty:
                    for emotion in list(emotions):
                        emo_df.at[i, emotion] += emo_score[emotion]

    new_df = pd.concat([new_df, emo_df], axis=1)

    return new_df


silmarillion_sentiments_final = text_emotion(silmarillion_sentiments, 'tweet')


# In[259]:


silmarillion_sentiments_final.head()


# In[260]:


silmarillion_sentiments_final['word_count'] = silmarillion_sentiments_final['Text'].apply(tokenize.word_tokenize).apply(len)
silmarillion_sentiments_final


# In[240]:


silmarillion_sentiments_final.Anger = silmarillion_sentiments_final.Anger.astype('float64') 
silmarillion_sentiments_final.Anticipation = silmarillion_sentiments_final.Anticipation.astype('float64') 
silmarillion_sentiments_final.Disgust = silmarillion_sentiments_final.Disgust.astype('float64') 
silmarillion_sentiments_final.Fear = silmarillion_sentiments_final.Fear.astype('float64') 
silmarillion_sentiments_final.Joy = silmarillion_sentiments_final.Joy.astype('float64') 
silmarillion_sentiments_final.Negative = silmarillion_sentiments_final.Negative_NRC.astype('float64') 
silmarillion_sentiments_final.Positive = silmarillion_sentiments_final.Positive_NRC.astype('float64') 
silmarillion_sentiments_final.Sadness = silmarillion_sentiments_final.Sadness.astype('float64') 
silmarillion_sentiments_final.Surprise = silmarillion_sentiments_final.Surprise.astype('float64') 
silmarillion_sentiments_final.Trust = silmarillion_sentiments_final.Trust.astype('float64') 
silmarillion_sentiments_final.word_count = silmarillion_sentiments_final.word_count.astype('float64') 


# In[261]:


silmarillion_sentiments_final.dtypes


# In[262]:


emotions = ['Positive_NRC','Negative_NRC','Anger','Anticipation','Disgust','Fear','Joy','Sadness','Surprise',
                 'Trust']


# In[263]:


for emotion in emotions:
    silmarillion_sentiments_final[emotion]=silmarillion_sentiments_final[emotion]/silmarillion_sentiments_final['word_count']

silmarillion_sentiments_final.head()


# In[264]:


silmarillion_sentiments_final.to_csv('./robinhood_emotions.csv')


# In[269]:


def ratings(silmarillion_sentiments_final):
    if silmarillion_sentiments_final['Compound'] > 0.05:
        return 1
    elif silmarillion_sentiments_final['Compound'] < -0.05:
        return -1
    else:
        return 0

silmarillion_sentiments_final['Rating_num'] = silmarillion_sentiments_final.apply(ratings, axis=1)

silmarillion_sentiments_final.to_csv(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\The_Silmarillion\silmarillion_sentiments.csv')

silmarillion_sentiments_final.head(10)
