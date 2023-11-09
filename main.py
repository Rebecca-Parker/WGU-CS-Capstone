
import pandas as pd
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import metrics
from sklearn import svm

datafile = "FoodDataTrainA.csv"

names = ['calories', 'vitaminPerCalorie', 'quality']
df = pd.read_csv(datafile, names=names)

#training with initial set, known nutrition, known quality

#logistic regression instead of linear regression because it is a classification problem
log_model = linear_model.LogisticRegression(max_iter=1000)
#svm model
svm_model = svm.SVC(max_iter=1000)


#dependent variable of model, the 'quality'
y1 = df.values[:, 2]
#independent variables of model, from nutrition label
x1 = df.values[:, 0:2]

#scaling the data to [0,1]
min_max_scaler = preprocessing.MinMaxScaler()
x1_with_minmax = min_max_scaler.fit_transform(x1)

#training the model
log_model.fit(x1_with_minmax, y1)
svm_model.fit(x1_with_minmax, y1)


#getting accuracy metrics for using first set to train, then test on first set
y1_true = y1
y1_pred_log = log_model.predict(x1_with_minmax)
print("accuracy of predicting original set 1, log")
print(metrics.accuracy_score(y1_true, y1_pred_log))
y1_pred_svm = svm_model.predict(x1_with_minmax)
print("accuracy of predicting original set 1, svm")
print(metrics.accuracy_score(y1_true, y1_pred_svm))



#testing with second set, known nutrition, known quality

datafile2 = "FoodDataTrainB.csv"
df2 = pd.read_csv(datafile2, names=names)
x2 = df2.values[:, 0:2]
x2_with_minmax = min_max_scaler.fit_transform(x2)
y2_true = df2.values[:, 2]
y2_pred_log = log_model.predict(x2_with_minmax)
print("accuracy of predicting on set 2, log")
print(metrics.accuracy_score(y2_true, y2_pred_log))
y2_pred_svm = svm_model.predict(x2_with_minmax)
print("accuracy of predicting on set 2, svm")
print(metrics.accuracy_score(y2_true, y2_pred_svm))



#predicting with third set, known nutrition, unknown quality

datafile3 = "FoodDataValidate.csv"
df3 = pd.read_csv(datafile3, names=names)
x3 = df3.values[:, 0:2]
x3_with_minmax = min_max_scaler.fit_transform(x3)
y3_true = df3.values[:, 2]
y3_pred_log = log_model.predict(x3_with_minmax)
print("accuracy of predicting on set 3, log")
print(metrics.accuracy_score(y3_true, y3_pred_log))
y3_pred_svm = svm_model.predict(x3_with_minmax)
print("accuracy of predicting on set 3, svm")
print(metrics.accuracy_score(y3_true, y3_pred_svm))



