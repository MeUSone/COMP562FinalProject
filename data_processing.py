import numpy as np
from nltk.tokenize import TweetTokenizer
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from nltk.corpus import stopwords
import string
import csv

def tokenlize(filename):
    #Read data
    data = pd.read_csv(filename)

    #Tokenize the main text
    tokenizer = TweetTokenizer(r"\w+")
    for a in range(len(data['text'])):
        data['text'][a] = tokenizer.tokenize(data['text'][a])

    #Get rid of stopwords and punctuation
    for a in range(len(data['text'])):
        data['text'][a] = list(filter(lambda i: i not in stopwords.words() and i not in list(string.punctuation), data['text'][a]))
    #See distribution
    #plt.hist(data['target'])
    #plt.show()

    #See MOST freqent word in Non-disaster posts
    frequent_word_non_disaster = defaultdict(lambda:0)
    frequent_word_disaster = defaultdict(lambda:0)
    for a in range(len(data)):
        if data['target'][a] == 0:
            for word in data['text'][a]:
                frequent_word_non_disaster[word] = frequent_word_non_disaster[word]+1
        else:
            for word in data['text'][a]:
                frequent_word_disaster[word] = frequent_word_disaster[word]+1
    frequent_word_disaster = {frequent_word_disaster: v for frequent_word_disaster, v in sorted(frequent_word_disaster.items(), key=lambda item: item[1],reverse=True)}
    frequent_word_non_disaster = {frequent_word_non_disaster: v for frequent_word_non_disaster, v in sorted(frequent_word_non_disaster.items(), key=lambda item: item[1],reverse=True)}
    print(list(frequent_word_non_disaster.items())[:100])
    print(list(frequent_word_disaster.items())[:100])
    plt.hist(list(frequent_word_non_disaster.items())[:10])
    plt.hist(list(frequent_word_disaster.items())[:10])
    plt.show()
    data.to_csv('processed_data.csv',index=False)



if __name__ == '__main__':
    tokenlize('/Users/jiayifu/Desktop/NLP2021Fall/COMP562_Final_Project/nlp-getting-started/train.csv')
