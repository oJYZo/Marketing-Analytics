# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 13:24:01 2020

@author: 14830
"""

import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
#%matplotlib qt
import numpy as np
import pandas as pd

from sklearn import preprocessing
from scipy.stats import zscore
from datetime import datetime
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,multilabel_confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier

from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import HashingVectorizer 


#load and look at data
reviews_data = pd.read_csv("C:\\Users\\14830\\Downloads\\reviews_annotated.csv", index_col = False)
reviews_data.head()

#We don't need string and numbers, but need to extra Stars numbers
reviews_data['stars'] = [int(i[:1]) for i in reviews_data.iloc[:,1]]
#Delete review_id and review_stars
reviews_data = reviews_data.drop(['review_id','review_stars'],axis=1)

corpus = reviews_data['review_text'].tolist()
len(corpus)

#Using hash vertorizer to extract 200 features
vec = HashingVectorizer(n_features = 200, norm = None)
vec_res = vec.fit_transform(corpus).toarray()
vec_df = pd.DataFrame(vec_res)
vec_df.head()


#Standardization:
mms = preprocessing.StandardScaler()
vec_df = mms.fit_transform(vec_df)
#vec_df = pd.DataFrame(vec_df_X)


#Update data: check the result of standardization
print(vec_df)



#%%% Split Data
#Divided Train & Validation & Test in 6:2:2
#First Divided Train & Test in 80:20, then divided Train & Validation in 75:25
reviews_X_train, reviews_X_test, reviews_y_train, reviews_y_test = train_test_split(vec_df,reviews_data['stars'], test_size = 0.2, random_state = 42)
reviews_X_train, reviews_X_validation, reviews_y_train, reviews_y_validation = train_test_split(reviews_X_train,reviews_y_train, test_size = 0.25, random_state = 42)


#%%% Train 
#Classifier -- KNN
#Create a empty dataframe to store accurancy
accuracy_df = pd.DataFrame(np.zeros((6,4)), columns=['Model', 'Train Accuracy','Validation Accuracy', 'Test Accuracy'])

trainScoreList = []
validScoreList = []


for k in range(1,31):
    # Simuate scores: returns vector, note cv argument
    clf_KNN = KNeighborsClassifier(n_neighbors = k)
    clf_KNN.fit(reviews_X_train, reviews_y_train)
    trainScore = clf_KNN.score(reviews_X_train, reviews_y_train)
    validScore = clf_KNN.score(reviews_X_validation, reviews_y_validation)
    # Append result to list
    trainScoreList.append(trainScore)
    validScoreList.append(validScore)
    print(k)

print('When K = %d, the highest train accuracy score is %.4f' % (np.argmax(trainScoreList) + 1, max(trainScoreList)))
print('When K = %d, the highest validation accuracy score is %.4f' % (np.argmax(validScoreList) + 1, max(validScoreList)))


reviews_pred_train  = clf_KNN.predict(reviews_X_train)
reviews_pred_validation = clf_KNN.predict(reviews_X_validation)


#Confusion_matrix(y_true, y_pred)
print("----- Train confusion matrix-----")
print(confusion_matrix(reviews_y_train, reviews_pred_train))
print("")
print("----- Validation confusion matrix-----")
print(confusion_matrix(reviews_y_validation, reviews_pred_validation))
print("")
#print("----- Test confusion matrix-----")
#print(confusion_matrix(reviews_y_test, reviews_pred_test))


#Accuracy measures
accuracy_df.iloc[0,0] = 'KNN'
accuracy_df.iloc[0,1] = accuracy_score(reviews_y_train, reviews_pred_train)
#accuracy_df.iloc[0,1]
accuracy_df.iloc[0,2] = accuracy_score(reviews_y_validation, reviews_pred_validation)
#accuracy_df.iloc[0,2]
#accuracy_df.iloc[0,3] = accuracy_score(reviews_y_test, reviews_pred_test)
#accuracy_df.iloc[0,3]



print("----- Train confusion matrix-----")
print(multilabel_confusion_matrix(reviews_y_train, reviews_pred_train))
print("")
print("----- Validation confusion matrix-----")
print(multilabel_confusion_matrix(reviews_y_validation, reviews_pred_validation))
print("")
#print("----- Test confusion matrix-----")
#print(multilabel_confusion_matrix(reviews_y_test, reviews_pred_test))

#Accuracy measures
accuracy_df.iloc[1,0] = 'KNN'
accuracy_df.iloc[1,1] = accuracy_score(reviews_y_train, reviews_pred_train)
#accuracy_df.iloc[1,1]
accuracy_df.iloc[1,2] = accuracy_score(reviews_y_validation, reviews_pred_validation)
#accuracy_df.iloc[1,2]
#accuracy_df.iloc[1,3] = accuracy_score(reviews_y_test, reviews_pred_test)
#accuracy_df.iloc[1,3]


# plot results
plt.plot(range(1,31), trainScoreList, label = "train score")
plt.plot(range(1,31), validScoreList, label = "test score")
plt.ylabel('Accuracy')
plt.xlabel('N_neighbors')
plt.legend()
plt.grid()


#%%% Summary the Rating of a review

clf_KNN = KNeighborsClassifier(n_neighbors = 18)
clf_KNN.fit(reviews_X_train, reviews_y_train)
print(clf_KNN.score(reviews_X_test, reviews_y_test))


print("Based on the results I got from the accuracy score, \
       when k = %d, it has the highest accuracy score with %.4f among all the predictions." % (np.argmax(validScoreList) + 1, max(validScoreList))) 



