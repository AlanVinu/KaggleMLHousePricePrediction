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

#MAE(Mean Absolute Error) Calculation
from sklearn.metrics import mean_absolute_error
predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)

#Validation data to be taken to determine the model's accuracy
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
#Model definition
melbourne_model = DecisionTreeRegressor()
#fitting
melbourne_model.fit(train_X, train_y)
#get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

#finding the sweet spot between overfitting and underfitting
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y) :
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
    
#comparing mae with different values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000] :
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes : %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes, my_mae))
    
#Improving prediction using RandomForest
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(random_state = 1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))