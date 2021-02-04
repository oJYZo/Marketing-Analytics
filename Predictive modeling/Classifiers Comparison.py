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

#%%% Q1

# load train_data, test_data and test_label
# print data shape
titanic_train_data = pd.read_csv("C:\\Users\\14830\\Downloads\\data_titanic_train.csv", index_col=0)
titanic_test_data = pd.read_csv("C:\\Users\\14830\\Downloads\\data_titanic_test.csv", index_col=0)
cancer_data = pd.read_csv("C:\\Users\\14830\\Downloads\\data_cancer_benigno_maligno.csv", index_col=0)
gender_submission = pd.read_csv("C:\\Users\\14830\\Downloads\\data_titanic_gender_submission.csv", index_col=0)


print(f'titanic_train_dat shape {titanic_train_data.shape} ')
print(f'titanic_test_data shape {titanic_test_data.shape} ')
print(f'cancer_data shape {cancer_data.shape} ')
print(f'gender_submission shape {gender_submission.shape} ')


#%%% Cancer
#For diagnosis   B=1 M=0
for i in range(cancer_data.shape[0]):
    if cancer_data.iloc[i,0] == 'B':
        cancer_data.iloc[i,0] = 1
    else:
        cancer_data.iloc[i,0] = 0
cancer_data['diagnosis'] = cancer_data['diagnosis'].astype("int")

cancer_data.describe()

#Check for if missing data exists
for column in cancer_data.columns:
    print(column,':', cancer_data[column].count()/len(cancer_data),'  ', cancer_data[column].count())
print()

cancer_data_y = cancer_data.diagnosis

#Standardization/Normalization
#Except label, other variable std
mms = preprocessing.StandardScaler()

cancer_data_x = mms.fit_transform(cancer_data.drop(columns='diagnosis'))
cancer_data = pd.DataFrame(cancer_data_x)

#Update data: check the result of standardization
cancer_data.info()


#%%% Split Data
#Divided Train & Validation & Test in 6:2:2
cancer_X_train, cancer_X_test, cancer_y_train, cancer_y_test = train_test_split(cancer_data, cancer_data_y, test_size = 0.2, random_state = 42)
cancer_X_train, cancer_X_validation, cancer_y_train, cancer_y_validation = train_test_split(cancer_X_train,
                                                                                             cancer_y_train, test_size = 0.25, random_state = 42)
print(f"train shape {cancer_X_train.shape}")
print(f"validation shape {cancer_X_validation.shape}")
print(f"test shape {cancer_X_test.shape}")


#%%% Train 
#a. KNN

accuracy_df = pd.DataFrame(np.zeros((5,4)), columns=['Model', 'Train Accuracy','Validation Accuracy', 'Test Accuracy'])

trainScoreList = []
validScoreList = []

for k in range(1,31):
    # Simuate scores: returns vector, note cv argument
    clf_KNN = KNeighborsClassifier(n_neighbors = k)
    clf_KNN.fit(cancer_X_train, cancer_y_train)
    trainScore = clf_KNN.score(cancer_X_train, cancer_y_train)
    validScore = clf_KNN.score(cancer_X_validation, cancer_y_validation)
    # Append result to list
    trainScoreList.append(trainScore)
    validScoreList.append(validScore)
    print(k)

print('When K = %d, the highest train accuracy score is %.4f' % (np.argmax(trainScoreList) + 1, max(trainScoreList)))
print('When K = %d, the highest validation accuracy score is %.4f' % (np.argmax(validScoreList) + 1, max(validScoreList)))


cancer_pred_train  = clf_KNN.predict(cancer_X_train)
cancer_pred_validation = clf_KNN.predict(cancer_X_validation)

#Confusion_matrix(y_true, y_pred)
print("----- Train confusion matrix-----")
print(confusion_matrix(cancer_y_train, cancer_pred_train))
print("")
print("----- Validation confusion matrix-----")
print(confusion_matrix(cancer_y_validation, cancer_pred_validation))
print("")
#print("----- Test confusion matrix-----")
#print(confusion_matrix(cancer_y_test, cancer_pred_test))

#Accuracy measures 
accuracy_df.iloc[0,0] = 'KNN'
accuracy_df.iloc[0,1] = accuracy_score(cancer_y_train, cancer_pred_train)
#accuracy_df.iloc[0,1]
accuracy_df.iloc[0,2] = accuracy_score(cancer_y_validation, cancer_pred_validation)
#accuracy_df.iloc[0,2]
#accuracy_df.iloc[0,3] = accuracy_score(cancer_y_test, cancer_pred_test)
#accuracy_df.iloc[0,3]


# plot results
plt.plot(range(1,31), trainScoreList, label = "train score")
plt.plot(range(1,31), validScoreList, label = "validation score")
plt.ylabel('Accuracy')
plt.xlabel('N_neighbors')
plt.legend()
plt.grid()



#%%% Train 
#b. Naive Bayes -- Gaussian Navie Bayes
clf_NB = GaussianNB()
clf_NB.fit(cancer_X_train, cancer_y_train)


cancer_pred_train  = clf_NB.predict(cancer_X_train)
cancer_pred_validation = clf_NB.predict(cancer_X_validation)

trainScore = clf_NB.score(cancer_X_train, cancer_y_train)
validScore = clf_NB.score(cancer_X_validation, cancer_y_validation)

print('The highest train accuracy score is %.4f' % (trainScore))
print('The highest validation accuracy score is %.4f' % (validScore))


#Confusion_matrix(y_true, y_pred)
print("----- Train confusion matrix-----")
print(confusion_matrix(cancer_y_train, cancer_pred_train))
print("")
print("----- Validation confusion matrix-----")
print(confusion_matrix(cancer_y_validation, cancer_pred_validation))
print("")
#print("----- Test confusion matrix-----")
#print(confusion_matrix(cancer_y_test, cancer_pred_test))


#Accuracy measures 
accuracy_df.iloc[1,0] = 'Naive Bayes'
accuracy_df.iloc[1,1] = accuracy_score(cancer_y_train, cancer_pred_train)
#accuracy_df.iloc[1,1]
accuracy_df.iloc[1,2] = accuracy_score(cancer_y_validation, cancer_pred_validation)
#accuracy_df.iloc[1,2]
#accuracy_df.iloc[1,3] = accuracy_score(cancer_y_test, cancer_pred_test)
#accuracy_df.iloc[1,3]


#%%% Train
#c. Lasso Regression
alphaList = [0.001, 0.01, 0.1, 1, 10, 100]

trainScoreList = []
validScoreList = []

for al in alphaList:
    # Simuate scores:  returns vector, note cv argument
    clf_lasso = linear_model.Lasso(al)
    clf_lasso.fit(cancer_X_train, cancer_y_train)
    trainScore = clf_lasso.score(cancer_X_train, cancer_y_train)
    validScore = clf_lasso.score(cancer_X_validation, cancer_y_validation)
    # Append result to list
    trainScoreList.append(trainScore)
    validScoreList.append(validScore)


print('When alpha = %d, the highest train accuracy score is %.4f' % (np.argmax(trainScoreList) + 1, max(trainScoreList)))
print('When alpha = %d, the highest validation accuracy score is %.4f' % (np.argmax(validScoreList) + 1, max(validScoreList)))

cancer_pred_train  = clf_lasso.predict(cancer_X_train)
cancer_pred_validation = clf_lasso.predict(cancer_X_validation)


#Confusion_matrix(y_true, y_pred)
print("----- Train confusion matrix-----")
print(confusion_matrix(cancer_y_train, cancer_pred_train > 0.5))
print("")
print("----- Validation confusion matrix-----")
print(confusion_matrix(cancer_y_validation, cancer_pred_validation > 0.5))
print("")
#print("----- Test confusion matrix-----")
#print(confusion_matrix(cancer_y_test, cancer_pred_test > 0.5))


#Accuracy measures 
accuracy_df.iloc[2,0] = 'Lasso Regression'
accuracy_df.iloc[2,1] = accuracy_score(cancer_y_train, cancer_pred_train > 0.5)
#accuracy_df.iloc[2,1]
accuracy_df.iloc[2,2] = accuracy_score(cancer_y_validation, cancer_pred_validation > 0.5)
#accuracy_df.iloc[2,2]
#accuracy_df.iloc[2,3] = accuracy_score(cancer_y_test, cancer_pred_test > 0.5)
#accuracy_df.iloc[2,3]


#%%% Train
#d. Regression Trees

param_grid ={'max_features':[2,3,4,5,6],'max_depth':[2,4,6,8,10],'n_estimators':[50, 100]}
# set up cross-validation shuffles
nmc = 1000
cvf = ShuffleSplit(n_splits = nmc, test_size = 0.2, random_state = 25)
# set up search
grid_search = GridSearchCV(RandomForestRegressor(random_state = 25), param_grid, cv=cvf, return_train_score = True)
# implement search
grid_search.fit(cancer_X_train, cancer_y_train)
# move results into DataFrame
results = pd.DataFrame(grid_search.cv_results_)
results = results[['rank_test_score','mean_train_score','mean_test_score','param_max_features','param_max_depth','param_n_estimators']]
best_param = results[results['rank_test_score']==1]


''' 
clf_DT = DecisionTreeClassifier(random_state = 0, max_depth = 7)
clf_DT.fit(cancer_X_train, cancer_y_train)

cancer_pred_train = cross_val_predict(clf_DT,cancer_X_train, cancer_y_train)
cancer_pred_validation = cross_val_predict(clf_DT,cancer_X_validation, cancer_y_validation)
cancer_pred_test = cross_val_predict(clf_DT,cancer_X_test, cancer_y_test)
'''


print('The highest mean validation accuarcy shows when max features = %.4f, max depth=%f and n_estimators is %s with accuracy = %d' %(best_param.iloc[0,3], best_param.iloc[0,5], best_param.iloc[0,4], best_param.iloc[0,2]))

print('When param_grid = %d, the highest train accuracy score is %.4f' % (np.argmax(param_grid) + 1, max(param_grid)))
print('When param_grid = %d, the highest validation accuracy score is %.4f' % (np.argmax(param_grid) + 1, max(param_grid)))


cancer_pred_train  = grid_search.predict(cancer_X_train)
cancer_pred_validation = grid_search.predict(cancer_X_validation)


#Confusion_matrix(y_true, y_pred)
print("----- Train confusion matrix-----")
print(confusion_matrix(cancer_y_train, cancer_pred_train))
print("")
print("----- Validation confusion matrix-----")
print(confusion_matrix(cancer_y_validation, cancer_pred_validation))
print("")
#print("----- Test confusion matrix-----")
#print(confusion_matrix(cancer_y_test, cancer_pred_test))


#Accuracy measures 
accuracy_df.iloc[3,0] = 'Regression Trees'
accuracy_df.iloc[3,1] = accuracy_score(cancer_y_train, cancer_pred_train)
#accuracy_df.iloc[3,1]
accuracy_df.iloc[3,2] = accuracy_score(cancer_y_validation, cancer_pred_validation)
#accuracy_df.iloc[3,2]
#accuracy_df.iloc[3,3] = accuracy_score(cancer_y_test, cancer_pred_test)
#accuracy_df.iloc[3,3]


#%%% Train
#e. Neural Nets
clf_NN = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10,5,3))

clf_NN = MLPClassifier()
clf_NN.fit(cancer_X_train, cancer_y_train)
trainScoreList = clf_NN.score(cancer_X_train, cancer_y_train)
validScoreList = clf_NN.score(cancer_X_validation, cancer_y_validation)


print('The highest train accuracy is %.4f' % trainScoreList)
print('The highest validation accuracy is %.4f' % validScoreList)

cancer_pred_train  = clf_NN.predict(cancer_X_train)
cancer_pred_validation = clf_NN.predict(cancer_X_validation)


print("----- Train confusion matrix-----")
print(confusion_matrix(cancer_y_train, cancer_pred_train))
print("")
print("----- Validation confusion matrix-----")
print(confusion_matrix(cancer_y_validation, cancer_pred_validation))
print("")
#print("----- Test confusion matrix-----")
#print(confusion_matrix(cancer_y_test, cancer_pred_test))


#Accuracy measures 
accuracy_df.iloc[4,0] = 'Neural Nets'
accuracy_df.iloc[4,1] = accuracy_score(cancer_y_train, cancer_pred_train)
#accuracy_df.iloc[4,1]
accuracy_df.iloc[4,2] = accuracy_score(cancer_y_validation, cancer_pred_validation)
#accuracy_df.iloc[4,2]
#accuracy_df.iloc[4,3] = accuracy_score(cancer_y_test, cancer_pred_test)
#accuracy_df.iloc[4,3]



#%%% Summary Cancer
# Accuracy score Comparison among all the predictive models
print(accuracy_df)

clf_KNN = KNeighborsClassifier(n_neighbors = 20)
clf_KNN.fit(cancer_X_train, cancer_y_train)
print(clf_KNN.score(cancer_X_test, cancer_y_test))



print("Conclusion: \
Based on the results I got from all of the accuracy scores, \
KNN = Neural Nets > Naive Bayes = Regression > Lasso Regression. \
KNN has the highest accuracy score, \
so in KNN, when k = %d, it has the highest test accurate result with %.4f among all the models in doing predictions." % (np.argmax(trainScoreList) + 1, max(trainScoreList)))



#%%% Titanic

#Look at data
titanic_train_data.describe()
titanic_test_data.describe()
gender_submission.describe()


#Check for if missing data exists in training sets and test sets
print('------Train Sets-------')
for column in titanic_train_data.columns:
    print(column,':',titanic_train_data[column].count()/len(titanic_train_data),'  ',titanic_train_data[column].count())
print()

print('------Test Sets-------')
for column in titanic_test_data.columns:
    print(column,':',titanic_test_data[column].count()/len(titanic_test_data),'  ',titanic_test_data[column].count())


#Drop Cabin & Ticket because it lost almost 80% data, and ticket is string
titanic_train_data = titanic_train_data.drop(['Cabin', 'Ticket'],axis = 1)
titanic_test_data = titanic_test_data.drop(['Cabin', 'Ticket'],axis = 1)


#Process Name and Extract Title
titanic_train_title = pd.DataFrame()
titanic_test_title = pd.DataFrame()

titanic_train_title["Title"] = titanic_train_data["Name"].map(lambda name:name.split(",")[1].split(".")[0].strip())
titanic_test_title["Title"] = titanic_test_data["Name"].map(lambda name:name.split(",")[1].split(".")[0].strip())

Title_Dictionary = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
}


titanic_train_title[ 'Title' ] = titanic_train_title.Title.map(Title_Dictionary)
titanic_test_title[ 'Title' ] = titanic_test_title.Title.map(Title_Dictionary)

titanic_train_title = pd.get_dummies(titanic_train_title.Title)
titanic_test_title = pd.get_dummies(titanic_test_title.Title)

titanic_train_data = pd.concat((titanic_train_data, titanic_train_title), axis=1)
titanic_test_data = pd.concat((titanic_test_data, titanic_test_title), axis=1)

titanic_train_data.pop("Name")
titanic_test_data.pop("Name")

print(f'titanic train_set shape {titanic_train_data.shape} ')
print(f'titanic test_set shape {titanic_test_data.shape} ')


#Fill missing value
mean_train = titanic_train_data.mean()
mean_test = titanic_test_data.mean()

titanic_train_data = titanic_train_data.fillna(mean_train)
titanic_test_data = titanic_test_data.fillna(mean_test)


#one-hot 
titanic_train_data["Pclass"] = titanic_train_data["Pclass"].astype(str)
titanic_test_data["Pclass"] = titanic_test_data["Pclass"].astype(str)
feature_dummies_train = pd.get_dummies(titanic_train_data[["Pclass", "Sex", "Embarked"]])
feature_dummies_test = pd.get_dummies(titanic_test_data[["Pclass", "Sex", "Embarked"]])

titanic_train_data.drop(["Pclass", "Sex", "Embarked"], inplace=True, axis=1)
titanic_test_data.drop(["Pclass", "Sex", "Embarked"], inplace=True, axis=1)

titanic_train_data = pd.concat((titanic_train_data, feature_dummies_train), axis=1)
titanic_test_data = pd.concat((titanic_test_data, feature_dummies_test), axis=1)


titanic_train_data_y = titanic_train_data.Survived

#Standardization/Normalization
mms = preprocessing.StandardScaler()

titanic_train_data_x = mms.fit_transform(titanic_train_data.drop(columns='Survived'))
titanic_train_data = pd.DataFrame(titanic_train_data_x)


titanic_test_data_x = mms.fit_transform(titanic_test_data)
titanic_test_data = pd.DataFrame(titanic_test_data_x)


#Update data: check the result of standardization
titanic_train_data.info()
titanic_test_data.info()



#%%% Split Data

titanic_X_train, titanic_X_validation, titanic_y_train, titanic_y_validation = train_test_split(titanic_train_data, titanic_train_data_y, test_size = 0.2, random_state = 42)
titanic_X_test = titanic_test_data  


print(f"train shape {titanic_X_train.shape}")
print(f"validation shape {titanic_X_validation.shape}")
print(f"test shape {titanic_X_test.shape}")



#%%% Train 
#a. KNN 
#Create a empty dataframe to store accuracy score
accuracy_df = pd.DataFrame(np.zeros((5,4)), columns = ['Model', 'Train Accuracy','Validation Accuracy', 'Test Accuracy'])

trainScoreList_tk = []
validScoreList_tk = []


for k in range(1,31):
    # Simuate scores: returns vector, note cv argument
    clf_KNN = KNeighborsClassifier(n_neighbors = k)
    #?cross_validate()
    clf_KNN.fit(titanic_X_train, titanic_y_train)
    trainScore = clf_KNN.score(titanic_X_train, titanic_y_train)
    validScore = clf_KNN.score(titanic_X_validation, titanic_y_validation)
    # Append result to list
    trainScoreList_tk.append(trainScore)
    validScoreList_tk.append(validScore)
    print(k)
    
print('When K = %d, the highest train accuracy score is %.4f' % (np.argmax(trainScoreList_tk) + 1, max(trainScoreList_tk)))
print('When K = %d, the highest validation accuracy score is %.4f' % (np.argmax(validScoreList_tk) + 1, max(validScoreList_tk)))


titanic_pred_train  = clf_KNN.predict(titanic_X_train)
titanic_pred_validation = clf_KNN.predict(titanic_X_validation)


#Confusion_matrix(y_true, y_pred)
print("----- Train confusion matrix-----")
print(confusion_matrix(titanic_y_train, titanic_pred_train))
print("")
print("----- Validation confusion matrix-----")
print(confusion_matrix(titanic_y_validation, titanic_pred_validation))
print("")
#print("----- Test confusion matrix-----")
#print(confusion_matrix(titanic_y_test, titanic_pred_test))


#Accuracy measures 
accuracy_df.iloc[0,0] = 'KNN'
accuracy_df.iloc[0,1] = accuracy_score(titanic_y_train, titanic_pred_train)
#accuracy_df.iloc[0,1]
accuracy_df.iloc[0,2] = accuracy_score(titanic_y_validation, titanic_pred_validation)
#accuracy_df.iloc[0,2]
#accuracy_df.iloc[0,3] = accuracy_score(titanic_y_test, titanic_pred_test)
#accuracy_df.iloc[0,3]



# plot results
plt.plot(range(1,31), trainScoreList, label = "train score")
plt.plot(range(1,31), validScoreList, label = "validation score")
plt.ylabel('Accuracy')
plt.xlabel('N_neighbors')
plt.legend()
plt.grid()



#%%% Train 
#b. Naive Bayes -- Gaussian Navie Bayes
clf_NB = GaussianNB()
clf_NB.fit(titanic_X_train, titanic_y_train)


titanic_pred_train  = clf_NB.predict(titanic_X_train)
titanic_pred_validation = clf_NB.predict(titanic_X_validation)

trainScore_tn = clf_NB.score(titanic_X_train, titanic_y_train)
validScore_tn = clf_NB.score(titanic_X_validation, titanic_y_validation)

print('The highest train accuracy score is %.4f' % (trainScore_tn))
print('The highest validation accuracy score is %.4f' % (validScore_tn))



#Confusion_matrix(y_true, y_pred)
print("----- Train confusion matrix-----")
print(confusion_matrix(titanic_y_train, titanic_pred_train))
print("")
print("----- Validation confusion matrix-----")
print(confusion_matrix(titanic_y_validation, titanic_pred_validation))
print("")
#print("----- Test confusion matrix-----")
#print(confusion_matrix(titanic_y_test, titanic_pred_test))



#Accuracy measures 
accuracy_df.iloc[1,0] = 'Naive Bayes'
accuracy_df.iloc[1,1] = accuracy_score(titanic_y_train, titanic_pred_train)
#accuracy_df.iloc[1,1]
accuracy_df.iloc[1,2] = accuracy_score(titanic_y_validation, titanic_pred_validation)
#accuracy_df.iloc[1,2]
#accuracy_df.iloc[1,3] = accuracy_score(titanic_y_test, titanic_pred_test)
#accuracy_df.iloc[1,3]



#%%% Train
#c. Lasso Regression

alphaList = [0.001, 0.01, 0.1, 1, 10, 100]
trainScoreList_tl = []
validScoreList_tl = []


for al in alphaList:
    # Simuate scores:  returns vector, note cv argument
    clf_lasso = linear_model.Lasso(al)
    clf_lasso.fit(titanic_X_train, titanic_y_train)
    trainScore = clf_lasso.score(titanic_X_train, titanic_y_train)
    validScore = clf_lasso.score(titanic_X_validation, titanic_y_validation)
    # Append result to list
    trainScoreList_tl.append(trainScore)
    validScoreList_tl.append(validScore)


print('When alpha = %d, the highest train accuracy score is %.4f' % (np.argmax(trainScoreList_tl) + 1, max(trainScoreList_tl)))
print('When alpha = %d, the highest validation accuracy score is %.4f' % (np.argmax(validScoreList_tl) + 1, max(validScoreList_tl)))

titanic_pred_train  = clf_lasso.predict(titanic_X_train)
titanic_pred_validation = clf_lasso.predict(titanic_X_validation)


#Confusion_matrix(y_true, y_pred)
print("----- Train confusion matrix-----")
print(confusion_matrix(titanic_y_train, titanic_pred_train >= 0.5))
print("")
print("----- Validation confusion matrix-----")
print(confusion_matrix(titanic_y_validation, titanic_pred_validation >= 0.5))
print("")
#print("----- Test confusion matrix-----")
#print(confusion_matrix(titanic_y_test, titanic_pred_test >= 0.5))


#Accuracy measures 
accuracy_df.iloc[2,0] = 'Lasso Regression'
accuracy_df.iloc[2,1] = accuracy_score(titanic_y_train, titanic_pred_train >0.5)
#accuracy_df.iloc[2,1]
accuracy_df.iloc[2,2] = accuracy_score(titanic_y_validation, titanic_pred_validation >0.5)
#accuracy_df.iloc[2,2]
#accuracy_df.iloc[2,3] = accuracy_score(titanic_y_test, titanic_pred_test >0.5)
#accuracy_df.iloc[2,3]


#%%% Train
#d. Regression Trees


param_grid_tt ={'max_features':[2,3,4,5,6],'max_depth':[2,4,6,8,10],'n_estimators':[50, 100]}
# set up cross-validation shuffles
nmc = 1000
cvf = ShuffleSplit(n_splits = nmc, test_size = 0.2, random_state = 25)
# set up search
grid_search = GridSearchCV(RandomForestRegressor(random_state = 25), param_grid_tt, cv=cvf, return_train_score = True)
# implement search
grid_search.fit(titanic_X_train, titanic_y_train)
# move results into DataFrame
results = pd.DataFrame(grid_search.cv_results_)
results = results[['rank_test_score','mean_train_score','mean_test_score','param_max_features','param_max_depth','param_n_estimators']]
best_param = results[results['rank_test_score']==1]



'''
clf_DT = DecisionTreeClassifier(random_state=0, max_depth = 7)
clf_DT.fit(titanic_X_train, titanic_y_train)

titanic_pred_train = cross_val_predict(clf_DT,titanic_X_train, titanic_y_train)
titanic_pred_validation = cross_val_predict(clf_DT,titanic_X_validation, titanic_y_validation)
'''

print('The highest mean validation accuarcy shows when max features = %.4f, max depth=%f and n_estimators is %s with accuracy = %d' %(best_param.iloc[0,3], best_param.iloc[0,5], best_param.iloc[0,4], best_param.iloc[0,2]))

print('When param_grid = %d, the highest train accuracy score is %.4f' % (np.argmax(param_grid_tt) + 1, max(param_grid_tt)))
print('When param_grid = %d, the highest validation accuracy score is %.4f' % (np.argmax(param_grid_tt) + 1, max(param_grid_tt)))


titanic_pred_train  = grid_search.predict(titanic_X_train)
titanic_pred_validation = grid_search.predict(titanic_X_validation)


#Confusion_matrix(y_true, y_pred)
print("----- Train confusion matrix-----")
print(confusion_matrix(titanic_y_train, titanic_pred_train))
print("")
print("----- Validation confusion matrix-----")
print(confusion_matrix(titanic_y_validation, titanic_pred_validation))
print("")
#print("----- Test confusion matrix-----")
#print(confusion_matrix(titanic_y_test, titanic_pred_test))


#Accuracy measures 
accuracy_df.iloc[3,0] = 'Regression Trees'
accuracy_df.iloc[3,1] = accuracy_score(titanic_y_train, titanic_pred_train)
#accuracy_df.iloc[3,1]
accuracy_df.iloc[3,2] = accuracy_score(titanic_y_validation, titanic_pred_validation)
#accuracy_df.iloc[3,2]


#%%% Train
#d. Neural Nets
# MLPClassifier
#(10,),(10,5),(10,5,3))
clf_NN = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10,5,3))
clf_NN.fit(titanic_X_train, titanic_y_train)
trainScoreList_tnn = clf_NN.score(titanic_X_train, titanic_y_train)
validScoreList_tnn = clf_NN.score(titanic_X_validation, titanic_y_validation)


print('The highest train accuracy is %.4f' % trainScoreList_tnn)
print('The highest validation accuracy is %.4f' % validScoreList_tnn)


titanic_pred_train  = clf_NN.predict(titanic_X_train)
titanic_pred_validation = clf_NN.predict(titanic_X_validation)


print("----- Train confusion matrix-----")
print(confusion_matrix(titanic_y_train, titanic_pred_train))
print("")
print("----- Validation confusion matrix-----")
print(confusion_matrix(titanic_y_validation, titanic_pred_validation))
print("")
#print("----- Test confusion matrix-----")
#print(confusion_matrix(titanic_y_test, titanic_pred_test))

#Accuracy measures 
accuracy_df.iloc[4,0] = 'Neural Nets'
accuracy_df.iloc[4,1] = accuracy_score(titanic_y_train, titanic_pred_train)
#accuracy_df.iloc[4,1]
accuracy_df.iloc[4,2] = accuracy_score(titanic_y_validation, titanic_pred_validation)
#accuracy_df.iloc[4,2]


#%%% Summary Titanic

print(accuracy_df)


clf_KNN = KNeighborsClassifier(n_neighbors = 9)
clf_KNN.fit(titanic_X_train, titanic_y_train)
print(clf_KNN.score(titanic_X_test, titanic_y_test))


print("Conclusion: \
Based on the results I got from all of the accuracy scores, \
KNN = Neural Nets > Regression Trees > Lasso Regression > Naive Bayes \
KNN has the highest accuracy score, \
so in KNN, when k = %d, \
it has the highest test accurate result with %.4f among all the models in doing predictions." % (np.argmax(trainScoreList) + 1, max(trainScoreList))


