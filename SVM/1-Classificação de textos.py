#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Created on 04-05-2020 / 19:49:01
    @author: Hiago dos Santos
    @e-mail: hiagop22@gmail.com
"""

import pandas as pd
import re
import numpy as np
from sklearn.pipeline import Pipeline # Cria filas de execução
from sklearn.feature_extraction.text import CountVectorizer # Verifica a freq de ocorrência no doc
from sklearn.feature_extraction .text import TfidfTransformer # Verifica a freq de ocorrência em outros doc
from sklearn.model_selection import train_test_split # separa os dados entre teste e validação
from sklearn.metrics import confusion_matrix, accuracy_score # Insere matrix de confusão e verif de score

# Algorithms
from sklearn.svm import LinearSVC, SVC

import matplotlib.pyplot as plt
import seaborn as sns

def clean_str(string):
    string = re.sub(r'\n', '', string)
    string = re.sub(r'\r', '', string)
    string = re.sub(r'[0-9]', 'digit', string)
    string = re.sub(r'\'', '*', string)
    string = re.sub(r'\"', '*', string)

    return string.strip().lower()

df = pd.read_csv("topics.csv") # Return de file paramater into a data_base
print(df.head()) # Return de first n rows

sns.countplot(y="question_topic", data=df)
plt.show()

X = [clean_str(df.iloc[x][1]) for x in range(df.shape[0])]
Y = np.array(df['question_topic'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SVC(kernel='rbf', gamma=0.08))
    # ('clf', LinearSVC(C=4.0)) c= tolerância anotada no caderno
])

model.fit(X_train, Y_train) # ajustar o modelo aos dados (treinar a rede)
pred = model.predict(X_test)
c_matrix = confusion_matrix(pred, Y_test) # Confusion matrix é uma matriz de erros 
sns.set(style='ticks', color_codes=True, rc={'figure.figsize': (12,8)},
        font_scale=1.2) # tupla com alt e larg para a fig
sns.heatmap(c_matrix, annot=True, annot_kws={'size': 12}) # mapa de calor, annot=True: mostra os números
                                                          #  de o quanto acertou e o quanto errou, tamanho
                                                          #  do texto que vai mostrar nas cel da matriz
plt.show()

print(accuracy_score(Y_test, pred)) # mostrar o score de acurácia

# testando com uma frase que não está no dataframe
text = 'Do you have this shirt in blue?'
text2 = 'Give me a descount'
print(model.predict([text][0]))

from sklearn.externals import joblib # exportar o modelo para usar depois sem precisar treinar
joblib.dump(model, 'model.pkl') #*.pkl sempre
model_imported = joblib.load('model.pkl')