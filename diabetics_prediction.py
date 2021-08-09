import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
#import plotly.plotly as py
from sklearn.preprocessing import scale
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, r2_score
#For LR
import statsmodels.api as sm
#For LR That looks like R
import statsmodels.formula.api as smf
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
print("Packages LOADED")
import os
print(os.getcwd())


os.chdir('C:\\Users\\keert\\Datasets')


data = pd.read_csv('diabetes2.csv')
data


data.info()


import sklearn
array = data.values
array


X = array[:,0:8] # ivs for train
y = array[:,8] # dv


test_size = 0.33


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size)


regr = skl_lm.LogisticRegression()
regr.fit(X_train, y_train)


pred = regr.predict(X_test)
pred


def confMtrx():
 from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
 cm_df = pd.DataFrame(confusion_matrix(y_test, pred).T, index=regr.classes_,
 columns=regr.classes_)
 cm_df.index.name = 'Predicted'
 cm_df.columns.name = 'True'
 print(cm_df)

 print(classification_report(y_test, pred))
 print(regr.score(X_test,y_test))
confMtrx()


joblib.dump(logreg,"model1")
