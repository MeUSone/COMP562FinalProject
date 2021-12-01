import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from ast import literal_eval

data = pd.read_csv('/Users/jiayifu/Desktop/NLP2021Fall/COMP562_Final_Project/processed_data.csv')

for a in range(len(data)):
    data['text'][a] = literal_eval(data['text'][a])
# See MOST freqent word in Non-disaster posts
frequent_word_non_disaster = defaultdict(lambda: 0)
frequent_word_disaster = defaultdict(lambda: 0)
numNonDe = 0
numDe = 0
for a in range(len(data)):
    if data['target'][a] == 0:
        for word in data['text'][a]:
            numNonDe += 1
            frequent_word_non_disaster[word] = frequent_word_non_disaster[word] + 1
    else:
        numDe += 1
        for word in data['text'][a]:
            frequent_word_disaster[word] = frequent_word_disaster[word] + 1
frequent_word_disaster = {frequent_word_disaster: v for frequent_word_disaster, v in
                          sorted(frequent_word_disaster.items(), key=lambda item: item[1], reverse=True)}
frequent_word_non_disaster = {frequent_word_non_disaster: v for frequent_word_non_disaster, v in
                              sorted(frequent_word_non_disaster.items(), key=lambda item: item[1], reverse=True)}
print(list(frequent_word_non_disaster.items())[:20])
print(list(frequent_word_disaster.items())[:20])
print(numDe)
print(numNonDe)
#None disaster posts
#[('...', 414), ('\x89', 412), ('like', 235), ("I'm", 192), ('Û_', 171), ('get', 143), ('via', 94), ('2', 93), ('would', 88),
# ('new', 87), ('know', 77), ('got', 77), ('people', 72), ('New', 72), ('3', 71), ('time', 70), ('Full', 70), ('body', 70), ('video', 70), ('day', 69)]
#Number 41318

#Disaster posts
#[('...', 633), ('\x89', 372), ('Û_', 171), ('fire', 127), ('via', 119), ('California', 104), ('people', 93), ('killed', 88),
# ('like', 87), ('2', 82), ('suicide', 76), ('fires', 74), ('disaster', 70), ('PM', 69), ('Hiroshima', 67), ('buildings', 67),
# ('crash', 66), ('MH370', 65), ('bomb', 61), ('Legionnaires', 61)]
#Number 3271
