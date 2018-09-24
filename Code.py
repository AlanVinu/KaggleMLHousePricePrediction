import os
import pandas as pd

#setting the working directory as the project's folder
os.chdir('C:/Users/Alan/Documents/Python/MLKaggleCourse')
#saving file path
melbourne_file_path = 'melb_data.csv'

#reading and storing data in DataFrame
melbourne_data = pd.read_csv(melbourne_file_path)

#summary
melbourne_data.describe()

#list of all columns
melbourne_data.columns

#dropping houses with missing values
melbourne_data = melbourne_data.dropna(axis = 0)

#prediction target is the Price
y = melbourne_data.Price

#selecting the features for the model
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
X.describe()

#Decision tree model
from sklearn.tree import DecisionTreeRegressor
#Defining model, a number is given for random_state to ensure same result in each run
melbourne_model = DecisionTreeRegressor(random_state = 1)
#fitting the model
melbourne_model.fit(X,y)

#testing the model by predicting for first 5 houses in the training data
print("Making predictions for the following 5 houses: ")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))