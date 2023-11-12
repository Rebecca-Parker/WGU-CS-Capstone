# WGU-CS-Capstone
November 2023
Rebecca Parker
Student ID 001333347

This is a simple machine learning algorithm to fulfill WGU Capstone requirements.

Initial problem: Baby food is typically sugar mush, with all the preservation and pureeing processes destroying the native vitamins. 
In order to give baby the best start with the best food, an application is needed to rate the quality of the baby food based on the vitamin content.

Dataset is from Kaggle, https://www.kaggle.com/datasets/shroukgomaa/babies-food-ingredients, accessed November 5, 2023.
Data has been preprocessed to sum all vitamin columns, calculate total vitamin per calories and remove all columns except calories and vitamin per calories.

Application rates baby food on a scale of 1-3, 1 being standard, 2 being better, 3 being best, based on vitamin content.

User can input calories per serving and either total vitamins per serving or individual vitamins per serving and application will rate the quality of the baby food.
