#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Created on 04-20-2020 / 12:36:43
    @author: Hiago dos Santos
    @e-mail: hiagop22@gmail.com
"""

from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(Features, classes)
# knn.predict(New_object)