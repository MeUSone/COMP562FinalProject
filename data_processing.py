import numpy as np
from nltk.tokenize import TweetTokenizer
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from nltk.corpus import stopwords
import string

def tokenlize(filename):
    #Read data
    data = pd.read_csv(filename)

    #Tokenize the main text
    tokenizer = TweetTokenizer(r"\w+")
    for a in range(len(data['text'])):
        data['text'][a] = tokenizer.tokenize(data['text'][a])

    #Get rid of stopwords and punctuation
    for a in range(len(data['text'])):
        data['text'][a] = list(filter(lambda i: i.lower() not in stopwords.words() and i.lower() not in list(string.punctuation), data['text'][a]))
    #See distribution
    #plt.hist(data['target'])
    #plt.show()

    data.to_csv('processed_data_test.csv',index=False,encoding='utf-8')



if __name__ == '__main__':
    #tokenlize('/Users/jiayifu/Desktop/NLP2021Fall/COMP562_Final_Project/nlp-getting-started/train.csv')
    tokenlize('/Users/jiayifu/Desktop/NLP2021Fall/COMP562_Final_Project/nlp-getting-started/test.csv')
