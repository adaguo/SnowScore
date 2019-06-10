import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

import collections

def tokenized(dataX, isStem = True, removeStopWord = True):
'''
list of strings (review)
'''
    tokenized = []
    porter = PorterStemmer()
    for text in dataX:
        # Discard non alpha-numeric characters
        # Set everything to lower case
        tokens = text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
        # Discard non-Egnlish words (not by default)
        # Stems all words using PorterStemmer, and change the stems back to the most occurring existent word. 
        # [https://www.datacamp.com/community/tutorials/stemming-lemmatization-python]
        if isStem or removeStopWord :
            for t in tokens:
                if isStem:
                    t = porter.stem(t)
            
        tokens = [porter.stem(t) if isStem else t for t in tokens if t not in stopwords.words('english')]
        tokenized.append(tokens)
        
    return tokenized

def transform(tokenized_train, tokenized_dev, tokenized_test):
    transformed_train = [" ".join(text) for text in tokenized_train]
    transformed_dev = [" ".join(text) for text in tokenized_dev]
    transformed_test = [" ".join(text) for text in tokenized_test]
    
    return transformed_train, transformed_dev, transformed_test
