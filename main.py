
import pandas as pd
import numpy as np
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import metrics
from sklearn import svm

datafile = "FoodDataTrainA1.csv"

names = ['calories', 'vitaminPerCalorie', 'quality']
df = pd.read_csv(datafile, names=names)

#dependent variable of model, the 'quality'
y1 = df.values[:, 2]
#independent variables of model, from nutrition label
x1 = df.values[:, 0:2]

#scaling the data to [0,1]
min_max_scaler = preprocessing.MinMaxScaler()
x1_with_minmax = min_max_scaler.fit_transform(x1)


#training with initial set, known nutrition, known quality
#svm model
svm_model = svm.SVC(max_iter=1000)


#training the model
svm_model.fit(x1_with_minmax, y1)


#getting accuracy metrics for testing on same set as training
y1_true = y1
y1_pred_svm = svm_model.predict(x1_with_minmax)
print("accuracy of predicting original set 1, svm")
print(metrics.accuracy_score(y1_true, y1_pred_svm))



#testing with second set, known nutrition, known quality

datafile2 = "FoodDataTrainB1.csv"
df2 = pd.read_csv(datafile2, names=names)
x2 = df2.values[:, 0:2]
x2_with_minmax = min_max_scaler.fit_transform(x2)
y2_true = df2.values[:, 2]
y2_pred_svm = svm_model.predict(x2_with_minmax)
print("accuracy of predicting on set 2, svm")
print(metrics.accuracy_score(y2_true, y2_pred_svm))



#predicting with third set, known nutrition, unknown quality

datafile3 = "FoodDataValidate1.csv"
df3 = pd.read_csv(datafile3, names=names)
x3 = df3.values[:, 0:2]
x3_with_minmax = min_max_scaler.fit_transform(x3)
y3_true = df3.values[:, 2]
y3_pred_svm = svm_model.predict(x3_with_minmax)
print("accuracy of predicting on set 3, svm")
print(metrics.accuracy_score(y3_true, y3_pred_svm))

new_calories = float(input("Please enter the calories per serving:\n"))
user_option = int(input("Please select an option: \n"
                        "Type 1 to enter total vitamins per serving \n"
                        "Type 2 to enter vitamin amounts individually \n"))

if user_option == 1:
    new_total_vitamin = float(input("Please enter the total vitamins per serving.\n"))
elif user_option == 2:
    new_vitA = float(input("Please enter the vit A(grams) per serving."))
    new_calcium = float(input("Please enter the calcium(mg) per serving."))
    new_thiamin = float(input("Please enter the thiamin(mg) per serving."))
    new_zinc = float(input("Please enter the zinc(mg) per serving."))
    new_potassium = float(input("Please enter the potassium(mg) per serving."))
    new_magnesium = float(input("Please enter the magnesium(mg) per serving."))
    new_vitE = float(input("Please enter the vit E(mg) per serving."))
    new_vitK = float(input("Please enter the vit K(mg) per serving."))
    new_vitC = float(input("Please enter the vit C(mg) per serving."))
    new_vitB6 = float(input("Please enter the vit B6(mg) per serving."))
    new_copper = float(input("Please enter the copper(mg) per serving."))
    new_carotene = float(input("Please enter the carotene(mg) per serving."))
    new_cryptoxanthene = float(input("Please enter the cryptoxanthene(mcg) per serving."))
    new_lycopene = float(input("Please enter the lycopene(mcg) per serving."))

    new_total_vitamin = (new_vitA +
                         new_calcium +
                         new_thiamin +
                         new_zinc +
                         new_potassium +
                         new_magnesium +
                         new_vitE +
                         new_vitK +
                         new_vitC +
                         new_vitB6 +
                         new_copper +
                         new_carotene +
                         new_cryptoxanthene +
                         new_lycopene)
else:
    print("I'm sorry Dave, I'm afraid I can't do that")

new_vitamin_per_calorie = new_total_vitamin / new_calories

if new_vitamin_per_calorie > 215.00:
    new_vitamin_per_calorie = 215.00
elif new_vitamin_per_calorie < 0.49:
    new_vitamin_per_calorie = 0.49

if new_calories > 53.76:
    new_calories = 53.76
elif new_calories < 3.33:
    new_calories = 3.33

new_prediction_data = [new_calories, new_vitamin_per_calorie]
print(new_prediction_data)

scaling_array = [[3.33, 0.49], [53.76, 215.79]]

scaling_array.append(new_prediction_data)

scaling_array_with_minmax = min_max_scaler.fit_transform(scaling_array)
scaled_prediction_data = scaling_array_with_minmax[-1]

predicted_quality = svm_model.predict([scaled_prediction_data])
print('The predicted quality for this item is: ')
print(predicted_quality)

