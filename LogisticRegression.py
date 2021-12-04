import pandas as pd
import gensim
from ast import literal_eval
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from numpy import  random
import multiprocessing


# Import data
df = pd.read_csv('/Users/jiayifu/Desktop/NLP2021Fall/COMP562_Final_Project/processed_data.csv')

for a in range(len(df)):
    df['text'][a] = literal_eval(df['text'][a])


def tagged_document(list_of_list_of_words):
    i = 0
    for list_of_words in list_of_list_of_words:
        i += 1
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])


train_documents = df['text']
train_documents = list(tagged_document(train_documents))
mcores = multiprocessing.cpu_count()

model_dbow = Doc2Vec(vector_size=300, workers=mcores)
model_dbow.build_vocab([x for x in tqdm(train_documents)])
model_dbow.train(train_documents, total_examples=len(df['text']), epochs=30)


def vector_for_learning(model, input_docs):
    sents = input_docs
    feature_vectors = [model.infer_vector(doc.words, steps=20) for doc in sents]
    return feature_vectors


model_dbow.save('./Model.d2v')

X_train = vector_for_learning(model_dbow, train_documents)

df2 = pd.read_csv('/Users/jiayifu/Desktop/NLP2021Fall/COMP562_Final_Project/processed_data_test.csv')
for a in range(len(df2)):
    df2['text'][a] = literal_eval(df2['text'][a])
# df2 = df2[df2['text'].apply(lambda x: len(x)) > 1]
test_documents = df2['text']
test_documents = list(tagged_document(test_documents))
X_test = vector_for_learning(model_dbow, test_documents)

random.seed(1000)

logreg = LogisticRegression()
logreg.fit(list(X_train), df['target'])
y_pred = logreg.predict(X_train)
y_test_pred = logreg.predict(X_test)

df1 = pd.DataFrame(list(zip((df['text']), list(df['target']), y_pred)), columns =['text', 'target', 'predict'])
df1.to_csv('result_train_Logistics.csv',index=False,encoding='utf-8')
print('Classification report: %s' % classification_report(df['target'], y_pred))
print('Accuracy score: %s' % accuracy_score(df['target'], y_pred))

df3 = pd.DataFrame(list(zip(df2['id'], y_test_pred)), columns =['id', 'target'])
df3.to_csv('result_test_Logistics.csv',index=False,encoding='utf-8')

# Classification report:               precision    recall  f1-score   support
#
#            0       0.70      0.85      0.77      4342
#            1       0.72      0.52      0.61      3271
#
#     accuracy                           0.71      7613
#    macro avg       0.71      0.69      0.69      7613
# weighted avg       0.71      0.71      0.70      7613

# 0.69384 for testing set

