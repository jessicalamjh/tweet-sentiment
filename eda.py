import pandas as pd

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import TweetTokenizer

import re

def load_dataset(url):
    # load dataset 
    dat = pd.read_csv(url)
    return pd.DataFrame(data=dat, columns=["text", "sentiment"])

def preprocess_data(df):
    # convert sentiment (categorical) to label (numerical)
    # negative = 0, neutral = 1, positive = 2
    df["sentiment"]  = df.sentiment.map({"negative": 0, "neutral": 1, "positive": 2}) 

    # remove missing values 
    df = df[df.text.notna()]
    return df

def get_wordnet_pos(tag):
    # renaming the POS tags to suit our needs
    if tag.startswith("J"):
        return wordnet.ADJ 
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return "" 

def get_stop_words():
    # get list of standard stop words
    stop_words = stopwords.words("english")

    # remove negations from stop_words (no, nor, not)
    del stop_words[116:119]

    # remove negative contractions from stop_words (e.g. couldn't)
    del stop_words[128:130]
    del stop_words[123:125]
    del stop_words[129:]

    # add neutral contractions to stop_words (e.g. they'll)
    stop_words.extend([
        "i'm", "i'll", "i'd", "us", "it'll", "it'd", 
        "they're", "they'd", "they'll", 
        "whose", "who're", "who\'ll", "who'd", 
        "that's"
    ])
    return stop_words
    
def _normalize(text, stop_words, lemmatizer=WordNetLemmatizer()):
    text = str(text).lower()
    
    no_twitterIDs = re.sub(r"@\S+", "", text)
    no_urls = re.sub(re.compile(r"http\S+|www\S+"), "", no_twitterIDs)
    no_numbers = re.sub("[0-9]+", "", no_urls)
    
    tokenizer = TweetTokenizer()
    tokenized = tokenizer.tokenize(no_numbers)
    
    re_words = re.compile(r"\w+")
    no_punctuation = [w for w in tokenized if re_words.search(w) and w not in stop_words]
    
    tagged = pos_tag(no_punctuation)
    
    lemmatized = []
    for word, tag in tagged:
        pos = get_wordnet_pos(tag)
        if pos:
            lemma = lemmatizer.lemmatize(word, pos)
        else:
            lemma = word
        lemmatized.append(lemma)
    return lemmatized

def normalize(df, stop_words):
    df["lemmatized"] = df.text.apply(lambda x: _normalize(x, stop_words))
    df.lemmatized = df.lemmatized.apply(lambda x: " ".join(x))
    return df