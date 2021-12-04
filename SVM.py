import pandas as pd
import gensim
from ast import literal_eval
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import svm
from numpy import random

# Import data
df = pd.read_csv('/Users/jiayifu/Desktop/NLP2021Fall/COMP562_Final_Project/processed_data.csv')


for a in range(len(df)):
    df['text'][a] = literal_eval(df['text'][a])

def tagged_document(list_of_list_of_words):
    i = 0
    for list_of_words in list_of_list_of_words:
        i += 1
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])

def vector_for_learning(model, input_docs):
    sents = input_docs
    feature_vectors = [model.infer_vector(doc.words, steps=20) for doc in sents]
    return feature_vectors

df2 = pd.read_csv('/Users/jiayifu/Desktop/NLP2021Fall/COMP562_Final_Project/processed_data_test.csv')
for a in range(len(df2)):
    df2['text'][a] = literal_eval(df2['text'][a])
# df2 = df2[df2['text'].apply(lambda x: len(x)) > 1]
test_documents = df2['text']
test_documents = list(tagged_document(test_documents))

train_documents = df['text']
train_documents = list(tagged_document(train_documents))

model_dbow = Doc2Vec.load('/Users/jiayifu/Desktop/NLP2021Fall/COMP562_Final_Project/Model.d2v')

X_train = vector_for_learning(model_dbow, train_documents)
X_test = vector_for_learning(model_dbow, test_documents)

random.seed(10)

clf = svm.SVC()
clf.fit(list(X_train), df['target'])

y_pred = clf.predict(list(X_train))
y_pred_test = clf.predict(list(X_test))

df1 = pd.DataFrame(list(zip((df2['id']), y_pred_test)), columns =['id', 'target'])
df1.to_csv('result_test_SVM.csv',index=False,encoding='utf-8')

print('Classification report: %s' % classification_report(df['target'], y_pred))
print('Accuracy score: %s' % accuracy_score(df['target'], y_pred))

# Classification report:               precision    recall  f1-score   support
#
#            0       0.74      0.92      0.82      4342
#            1       0.84      0.57      0.68      3271
#
#     accuracy                           0.77      7613
#    macro avg       0.79      0.74      0.75      7613
# weighted avg       0.78      0.77      0.76      7613
#
# Accuracy score: 0.766846184158676

# 0.72663 for testing set

