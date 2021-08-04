#https://github.com/NBrisbon/Silmarillion-NLP/blob/master/Sentiment_Analysis.py

import pandas as pd
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm_notebook as tqdm
from tqdm import trange

 df = pd.read_csv ('./0_results_robinhood.csv', index_col=None, header=0)

def text_emotion(text, column):
    '''
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame with ten new columns for each emotion
    '''
   
    new_df = df.copy()

    xlsx = pd.read_excel('./robinhood/cleaned_data.xlsx')
    emolex_df = xlsx[['text', 'Positive_NRC','Negative_NRC','Anger', 'Anticipation', 'Disgust', 'Fear','Joy',
                      'Sadness', 'Surprise', 'Trust']]
    emotions = emolex_df.columns.drop('text')
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


df = text_emotion(df, 'text')


# In[259]:


df.head()


# In[260]:


df['word_count'] = df['text'].apply(tokenize.word_tokenize).apply(len)
df


# In[240]:


df.Anger = df.Anger.astype('float64') 
df.Anticipation = df.Anticipation.astype('float64') 
df.Disgust = df.Disgust.astype('float64') 
df.Fear = df.Fear.astype('float64') 
df.Joy = df.Joy.astype('float64') 
df.Negative = df.Negative_NRC.astype('float64') 
df.Positive = df.Positive_NRC.astype('float64') 
df.Sadness = df.Sadness.astype('float64') 
df.Surprise = df.Surprise.astype('float64') 
df.Trust = df.Trust.astype('float64') 
df.word_count = df.word_count.astype('float64') 


# In[261]:


df.dtypes


# In[262]:


emotions = ['Positive_NRC','Negative_NRC','Anger','Anticipation','Disgust','Fear','Joy','Sadness','Surprise',
                 'Trust']


# In[263]:


for emotion in emotions:
    df[emotion]=df[emotion]/df['word_count']

df.head()


# In[264]:


df.to_csv('./robinhood_emotions.csv')


# In[269]:


def ratings(df):
    if df['Compound'] > 0.05:
        return 1
    elif df['Compound'] < -0.05:
        return -1
    else:
        return 0

df['Rating_num'] = df.apply(ratings, axis=1)

df.to_csv('./robinhood_emotions.csv')

df.head(10)
