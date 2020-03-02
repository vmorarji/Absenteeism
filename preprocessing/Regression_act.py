import numpy as np
import pandas as pd

data_preprocessed = pd.read_csv('df_preprocessed.csv')


## targets will be where absenteeism is greater than 3 hours(the median)

targets = np.where(data_preprocessed['Absenteeism Time in Hours']>
                   data_preprocessed['Absenteeism Time in Hours'].median(),1,0)



data_preprocessed['Excessive Absenteeism'] = targets



data_with_targets = data_preprocessed.drop(['Absenteeism Time in Hours','Daily Work Load Average', 'Day of the Week', 'Distance to Work'], axis=1)



## create table without the target
unscaled_inputs = data_with_targets.iloc[:,:-1]



## create a custom scaler to only scale the non-dummy variables
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class CustomScaler(BaseEstimator,TransformerMixin):
    
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns],y)
        self.mean_= np.mean(X[self.columns])
        self.var_= np.var(X[self.columns])
        return self
    
    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]

unscaled_inputs.columns.values



columns_to_omit = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Education']
columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]


absenteeism_scaler = CustomScaler(columns_to_scale)
absenteeism_scaler.fit(unscaled_inputs)
scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)


## split the data into training and testing sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size = 0.8, random_state = 20)


## Create and run the logistic regression

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

reg = LogisticRegression()
reg.fit(X_train,y_train)
model_outputs = reg.predict(X_train)


## create a summary table with the coeffiecients and the odds ratio
feature_name = unscaled_inputs.columns.values
summary_table = pd.DataFrame(columns = ['Feature Name'], data = feature_name)
summary_table['Coefficient'] = np.transpose(reg.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table['Odds_ratio'] = np.exp(summary_table.Coefficient)

summary_table.sort_values('Odds_ratio', ascending=False)


reg.score(X_test,y_test)


##pickle the model and the scaler

import pickle

with open('model', 'wb') as file:
    pickle.dump(reg, file)


with open('scaler','wb') as file:
    pickle.dump(absenteeism_scaler, file)
