
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn import preprocessing
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import ConfusionMatrixDisplay

# Initial training data
datafile = "FoodDataTrain.csv"
names = ['calories', 'vitaminPerCalorie', 'quality']
df = pd.read_csv(datafile, names=names)

# Visualizations for the data
num_qualities = df.groupby(by='quality').size()
print('')
print('Number of each quality category\n')
print(num_qualities)
print('')

calories_histoplot = sns.histplot(df, x='calories', hue='quality', kde=True, bins=30)
plt.show()

vitamins_histoplot = sns.histplot(df, x='vitaminPerCalorie', hue='quality', kde=True, bins=30)
plt.show()

data_visual_scatter = sns.pairplot(df, hue='quality')
plt.show()

# Dependent variable of model, the 'quality'
y1 = df.values[:, 2]
# Independent variables of model, from nutrition label
x1 = df.values[:, 0:2]

# Scaling the data to [0,1]
min_max_scaler = preprocessing.MinMaxScaler()
x1_with_minmax = min_max_scaler.fit_transform(x1)


# Training with initial set, known nutrition, known quality
# SVM model
svm_model = svm.SVC(max_iter=1000)
svm_model.fit(x1_with_minmax, y1)


# Getting accuracy metrics for testing on same set as training
y1_true = y1
y1_pred_svm = svm_model.predict(x1_with_minmax)
print("\nAccuracy of predicting original set 1, same set as training, using svm:")
print(metrics.accuracy_score(y1_true, y1_pred_svm))
print("See first confusion matrix")
ConfusionMatrixDisplay.from_predictions(y1_true, y1_pred_svm)
plt.show()


# Testing with second set, known nutrition, known quality
datafile2 = "FoodDataValidate.csv"
df2 = pd.read_csv(datafile2, names=names)
x2 = df2.values[:, 0:2]
x2_with_minmax = min_max_scaler.fit_transform(x2)
y2_true = df2.values[:, 2]
y2_pred_svm = svm_model.predict(x2_with_minmax)
print("\nAccuracy of prediction on set 2, new set, known quality, svm:")
print(metrics.accuracy_score(y2_true, y2_pred_svm))
print("See second confusion matrix")
ConfusionMatrixDisplay.from_predictions(y2_true, y2_pred_svm)
plt.show()


# Get new data to predict with, known nutrition, unknown quality, user input
new_calories = float(input("Please enter the calories per serving:\n"))
if new_calories <= 0:
    new_calories = 0.01

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
    print("Please start again")
    quit()

new_vitamin_per_calorie = new_total_vitamin / new_calories

# Each parameter needs to be within the bounds of the training set.
# There is no qualitative difference to any values above or below so this will bring the new values within bounds.
if new_vitamin_per_calorie > 215.00:
    new_vitamin_per_calorie = 215.00
elif new_vitamin_per_calorie < 0.49:
    new_vitamin_per_calorie = 0.49

if new_calories > 53.76:
    new_calories = 53.76
elif new_calories < 3.33:
    new_calories = 3.33

new_prediction_data = [new_calories, new_vitamin_per_calorie]
# print(new_prediction_data) # test statement

# The training array was scaled using a particular min and max.
# To keep the new data consistent, it needs to be scaled the same way so this
# array provides the previously used min and max.
scaling_array = [[3.33, 0.49], [53.76, 215.79]]
# Add the new data
scaling_array.append(new_prediction_data)
# Scale the new data so it aligns with how the model was trained
scaling_array_with_minmax = min_max_scaler.fit_transform(scaling_array)
# Retrieve just the new scaled data
scaled_prediction_data = scaling_array_with_minmax[-1]
# Get the model's prediction off of the new scaled input
predicted_quality = svm_model.predict([scaled_prediction_data])
print('The predicted quality for this item is: ')
print(predicted_quality)
print('1 = standard, 2 = good, 3 = best')
