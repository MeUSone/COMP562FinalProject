import numpy as np
from nltk.tokenize import TweetTokenizer
import pandas as pd
import matplotlib.pyplot as plt

def tokenlize(filename):
    #Read data
    data = pd.read_csv(filename)
    print(data['text'][3])

    #Tokenize the main text
    tokenizer = TweetTokenizer()
    for a in range(len(data['text'])):
        data['text'][a] = tokenizer.tokenize(data['text'][a])
    print(data['text'][3])

    #See distribution
    plt.hist(data['target'])
    plt.show()



if __name__ == '__main__':
    tokenlize('/Users/jiayifu/Desktop/NLP2021Fall/COMP562_Final_Project/nlp-getting-started/train.csv')
