"""
This script demonstrates the k-nearest neighbours (knn) algorithm 
on one of the freely available datasets used for machine learning algorithm testing.
The script includes simple knn algorithm implementation and compares its result
with the scikit-learn library implementation
"""

import pandas as pd
import numpy as np
from collections import Counter
from scipy.linalg import norm

from sklearn import model_selection

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

"""
dataset info:    
The dataset contains cases from a study that was conducted between
1958 and 1970 at the University of Chicago's Billings Hospital on
the survival of patients who had undergone surgery for breast
cancer. 
http://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival
"""
names = ['age at operation', 'year of operation', 'no. of detected nodes', 'class']
dataset = pd.read_csv('haberman.data.txt', names=names)

# getting the attributes and corresponding classes
array = dataset.values
X = array[:,0:3]
Y = array[:,3]

validation_size = 0.20 # we want to select 20 percent of dataset for validation
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)    


# x - data point/attribute used for prediction/classification
# x_train - data points/attributes which are a priori known or classified
# y_train - classes of the x_train data
# K - number of the nearest neighbours which will be used for classification/prediction ... needs to be an odd number for majority classifier
def knn_predict(x,x_train,y_train,K=3):
    ds=np.zeros_like(x_train[:,0])
    for k in range(len(x_train[:,0])):   
        ds[k]=norm(x_train[k,:]-x) # calculation of Frobenius norm which in the case of vector is same as Euclidean distance
    return Counter(y_train[np.argsort(ds)[:K]]).most_common(1)[0][0] # order by distance, select the first K and return the most common class


# classification of validation data
preds=np.zeros_like(X_validation[:,0])
spred=0
for k in range(len(X_validation[:,0])): 
    preds[k]=knn_predict(X_validation[k,:],X_train,Y_train,5)
    if preds[k]==Y_validation[k]:
        spred=spred+1
        
print('Accuracy of self-made knn is: '+str(spred/len(X_validation[:,0])*100))

# using of scikit-learn knn algorithm for comparison
knn = KNeighborsClassifier(n_neighbors=5, p=2) # p=2 for Euclidean distance
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)

print('Accuracy of scikit-learn knn is: '+str(accuracy_score(Y_validation, predictions)))

