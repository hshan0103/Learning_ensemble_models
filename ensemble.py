# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 20:22:52 2020

@author: Hshan
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import KFold
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
#import os 
import warnings
warnings.filterwarnings('ignore')
import time

fnc = pd.read_csv('C:/Users/Hshan/Desktop/neuroimaging/fnc.csv')
loading = pd.read_csv('C:/Users/Hshan/Desktop/neuroimaging/loading.csv')
train_scores = pd.read_csv('C:/Users/Hshan/Desktop/neuroimaging/train_scores.csv')


################################### defined function ##################################################
def metrics(y, y_hat):
    score = np.mean(np.sum(np.abs(y - y_hat), axis = 0)/np.sum(y_hat, axis = 0))
    return (score)

def load_data(is_train=True):

    fnc_cols = list(fnc.columns)
    fnc_cols.remove('Id')
    fnc[fnc_cols] *=0.0035
    # rescaling fnc dataframe

    features_df = fnc.merge(loading, on = 'Id')
    train_scores['is_train'] = True
    df = features_df.merge(train_scores, how='left')

    df['is_train'].fillna(False, inplace = True)
    
    if not is_train:
        df = df[df['is_train']==False]
        df.drop('is_train', axis=1, inplace= True)
    else:
        df = df[df['is_train']==True]
        df.drop('is_train', axis=1, inplace= True)
    
    return df

def train_predict(train_df, test_df, n=7):

    prediction = pd.DataFrame(test_df['Id'])
    tot_scores = 0
    ridge_tot_scores = 0

    kf = KFold(n_splits=n, shuffle=True, random_state=1231)

    for label in label_cols:
        scores = 0
        pred = 0
        ridge_pred = 0
        ridge_scores = 0
    
        lb, ub = lower[label], upper[label]
    
        null_df = train_df[train_df[label].isnull()]
    
        tr_df = train_df.loc[(train_df[label]>lb) & (train_df[label]<ub)]
    
        tr_df = tr_df.append(null_df)
    
        fill = 0.5*(tr_df[label].mean() + tr_df[label].median())
        tr_df = tr_df.fillna(fill)
    
        x = tr_df[features_cols]
        y= tr_df[label]
    
        for train_index, test_index in kf.split(tr_df):
            xTrain, xTest = x.iloc[train_index], x.iloc[test_index]
            yTrain, yTest = y.iloc[train_index], y.iloc[test_index]
     
            model = SVR(C=c[label], cache_size=3000, gamma=g[label], epsilon=3.5)
            model.fit(xTrain, yTrain)
            y_hat = model.predict(xTest)
    
            scores += metrics(y=yTest, y_hat = y_hat)
        
            #ridge
            ridge = Ridge(alpha=alpha[label], normalize=True)
            ridge.fit(xTrain, yTrain)
            ridge_y_hat = ridge.predict(xTest)

            ridge_scores += metrics(y=yTest, y_hat = ridge_y_hat)
            
            pred += model.predict(test_df[features_cols])
            ridge_pred += ridge.predict(test_df[features_cols])   
    
        prediction[f'ridge_{label}'] = ridge_pred/n
        prediction[label] = pred/n
        tot_scores += weights[label]*(scores/n)
        ridge_tot_scores += weights[label]*(ridge_scores/n)
    
        print(label, ': ', scores/n)    
        print(label, ': ', ridge_scores/n)

    print('overall score: ', round(tot_scores,5))
    print('ridge overall score: ', round(ridge_tot_scores,5))

    return prediction

def submission(prediction):
    for label in label_cols:
        prediction[label] = 0.7*prediction[label]+0.3*prediction[f'ridge_{label}']
        prediction.drop(f'ridge_{label}', axis=1, inplace=True)
    
    prediction= pd.melt(prediction, id_vars =['Id'], value_vars = label_cols) 

    prediction['Id'] = prediction['Id'].astype('str') + '_' + prediction['variable'].astype('str') 
    prediction.drop('variable', axis = 1, inplace = True)
    
    return prediction

def save(prediction, path):
    prediction.rename(columns = {'value':'Predicted'}, inplace = True)
    prediction.to_csv(path, index=False)
    
#######################################################################################################
    

train_df = load_data(is_train=True)
train_df.shape

test_df = load_data(is_train=False)
test_df.shape

for col in list(train_df.columns):
    if any(train_df[col].isnull()):
        print(col, 'null')
        
label_cols = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
features_cols = list(train_df.columns)
removed = label_cols + ['Id']
features_cols = [col for col in features_cols if col not in removed]

plt.show()
for label in label_cols:
    plt.hist(train_df[label],50)
    
sns.heatmap(loading.corr(),annot=True,linewidths=0.2) 
fig=plt.gcf()
fig.set_size_inches(20,10)

sns.heatmap(train_scores.corr(),annot=True,linewidths=0.2) 
fig=plt.gcf()
fig.set_size_inches(20,12)

# heatmap doesnt show much correlation worth extra attention
# no null for features
# null present for target variables

# strategy:
        # from the plot of the target variables, no obvious skewness spotted
        # outliers present, except for AGE
        # since the number of outliers are relatively smaller as compared to the sample size of trian dataframe
        # opted to remove outliers, rather than setting it to the max of the boundries 
        # as the number of null values in target variables is around 7% of the entire sample
        # the sample size of 5877 is not too big, opted to do imputation instead of dropping NA
        # outliers had been removed, impute the missing values with mean of that particular column
        # in theory, median is more robist to outliers, may consider to impute the missing value with median and keep the outliers 
        

# weights given in the instruction for the error contributed by each target variable
weights = {
            'age': 0.3, 
            'domain1_var1': 0.175, 
            'domain1_var2': 0.175, 
            'domain2_var1': 0.175, 
            'domain2_var2': 0.175
            }

g = {
            'age': 100, 
            'domain1_var1': 100, 
            'domain1_var2': 100, 
            'domain2_var1': 100, 
            'domain2_var2': 20
            }

c = {
            'age': 60, 
            'domain1_var1': 5, 
            'domain1_var2': 5, 
            'domain2_var1': 10, 
            'domain2_var2': 10
            }

# compute the limit for points to be identified as outliers 
# by computing lower=Q1-1.5*IQR, upper=Q3+1.5*IQR
# Example on computation:
    # Q1 = np.quantile(train_df[label], 0.25)
    # Q3 = np.quantile(train_df['age'], 0.75)
    # IQR = Q3-Q1
lower = {
            'age': 15.60401 , 
            'domain1_var1': 24.20974, 
            'domain1_var2': 22.66253, 
            'domain2_var1': 22.37824 , 
            'domain2_var2': 21.68834 
            }

upper = {
            'age': 87.67281, 
            'domain1_var1': 79.06708, 
            'domain1_var2': 80.61429  , 
            'domain2_var1': 80.89857  , 
            'domain2_var2': 81.58847 
            }

# parameter for ridge regressor 
alpha = {
            'age':0.5, 
            'domain1_var1': 2.5, 
            'domain1_var2': 2.5, 
            'domain2_var1': 2.5, 
            'domain2_var2': 2.5
            }
start = time.time()

prediction = train_predict(train_df, test_df, n=7)
submission(prediction)

stop = time.time()

print('Time: ', stop - start)  

save(prediction,'C:/Users/Hshan/Desktop/neuroimaging/prediction_ver65.csv')
