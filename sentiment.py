import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

## this turns words into their base forms
## for example, dogs & dog become one word
wordnet_lemmatizer = WordNetLemmatizer()

## set up the stopwords (words you wanna avoid)
stops = set(w.rstrip() for w in stopwords.words('english'))

## load up the positive reviews
positive_reviews = BeautifulSoup(open('data/electronics/positive.review').read(),
                                 features='lxml')

## filter the XML values to find the tag "review_text"
positive_reviews = positive_reviews.findAll('review_text')

## do the same for the negative reviews
negative_reviews = BeautifulSoup(open('data/electronics/negative.review').read(),
                                 features='lxml')
negative_reviews = negative_reviews.findAll('review_text')

## right now there are more positive then negative reviews, so we'll
## shuffle the positive reviews and keep a subset of positive reviews that is 
## as big as the negative reviews
np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]

## set up so that we can index all the words for ease of use later
word_index_map = {}
current_index = 0

def my_tokenizer(s):
    """
    tokenize a string s for fitting to logistic regression
    """
    s = s.lower()
    ## use nltk's tokenizer 
    tokens = nltk.tokenize.word_tokenize(s)
    ## only keep words that have more then 2 characters 
    tokens = [t for t in tokens if len(t)>2]
    ## only use the base form of words (e.g. jumping -> jump, dogs -> dog)
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    ## remove stop words
    tokens = [t for t in tokens if t not in stops]
    return tokens

for review in positive_reviews:
    tokens = my_tokenizer(review.text)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1
            
            
            
            
            
            
            