import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
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

## set up so that we can index all the words for ease of use later
word_index_map = {}
current_index = 0

# initialize to store the tokenized versions of the positive & negative reviews
positive_tokenized = []
negative_tokenized = []

for review in positive_reviews:
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1
            
for review in negative_reviews:
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1            
            
## create an array from the tokens
def tokens_to_vector(tokens, label):
    """
    Turn a tokenized array into a feature vector based on normalized counts
    """
    # initalize. the +1 is for the label
    x = np.zeros(len(word_index_map) + 1)  

    ## now do the actual word counts
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    ## normalize
    x = x/x.sum()
    x[-1] = label 
    return x

N = len(positive_tokenized) + len(negative_tokenized)


data = np.zeros((N, len(word_index_map) + 1))
i = 0
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1)
    data[i,:] = xy
    i += 1
    
for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0)
    data[i,:] = xy
    i += 1    
         
## data now contains our feature vectors
## shuffle them then split into train & test as usual
np.random.shuffle(data)

X = data[:, :-1] # everything except the last column
Y = data[:, -1] # only the last column

# train on all data minus the last 100
Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

## fit the model on the training data
model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print("Classification rate: ", model.score(Xtest, Ytest))






            
            
            