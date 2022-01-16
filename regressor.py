# %%
from numpy.random.mtrand import RandomState
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("Fish.csv")

# Data Parameters:
  # Length 1 = Vertical length in centimeters
  # Legnth 2 = Diagonal length in centimeters
  # Length 3 = Cross length in centimeters
  # Height = Height in centimeters
  # Width = Diagonal width in centimeters

# We will split the dataset into the continuous input variables vs continuous output variables
  # x will represent our inputs
  # y will represent out outputs

input_features = ['Length1', 'Length2', 'Length3', 'Height', 'Width']
x = df[input_features]
y = df.Weight

# Train test split will split 80% for training and 20% of the data for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# We will experiment with three different combinations of parameters and compare them in a table in the report
model_one = MLPRegressor(hidden_layer_sizes = (10, 10, 10, 10, 10, 10, 10), solver = 'adam', max_iter = 100000)
model_two = MLPRegressor(hidden_layer_sizes = (1000, 500, 250, 125, 60), solver = 'adam', max_iter = 100000)
model_three = MLPRegressor(hidden_layer_sizes = (1000, 1000, 1000), solver = 'adam', max_iter = 100000)

# We then train the data on each model 
model_one.fit(x_train, y_train)
model_two.fit(x_train, y_train)
model_three.fit(x_train, y_train)

# Model one: hidden_layer_sizes = 100, solver = 'adam', max_iter = 100000
ymodel_one_train = model_one.predict(x_train)
ymodel_one_test = model_one.predict(x_test)

# Model two: hidden_layer_sizes = 150, solver = 'adam', max_iter = 100000
ymodel_two_train = model_two.predict(x_train)
ymodel_two_test = model_two.predict(x_test)

# Model three: hidden_layer_sizes = 200, solver = 'adam', max_iter = 100000
ymodel_three_train = model_three.predict(x_train)
ymodel_three_test = model_three.predict(x_test)

# Model one training data
print("Model one training data:")
print("Mean Squared Error: ", mean_squared_error(y_train, ymodel_one_train))
print("Mean Absolute Error: ", mean_absolute_error(y_train, ymodel_one_train))
print("R2 Score: ", r2_score(y_train, ymodel_one_train), '\n')

# Model one test data
print("Model one test data:")
print("Mean Squared Error: ", mean_squared_error(y_test, ymodel_one_test))
print("Mean Absolute Error: ", mean_absolute_error(y_test, ymodel_one_test))
print("R2 Score: ", r2_score(y_test, ymodel_one_test), '\n')

# Model two training data
print("Model two training data:")
print("Mean Squared Error: ", mean_squared_error(y_train, ymodel_two_train))
print("Mean Absolute Error: ", mean_absolute_error(y_train, ymodel_two_train))
print("R2 Score: ", r2_score(y_train, ymodel_two_train), '\n')

# Model two test data
print("Model two test data:")
print("Mean Squared Error: ", mean_squared_error(y_test, ymodel_two_test))
print("Mean Absolute Error: ", mean_absolute_error(y_test, ymodel_two_test))
print("R2 Score: ", r2_score(y_test, ymodel_two_test), '\n')

# Model three training data
print("Model three training data:")
print("Mean Squared Error: ", mean_squared_error(y_train, ymodel_three_train))
print("Mean Absolute Error: ", mean_absolute_error(y_train, ymodel_three_train))
print("R2 Score: ", r2_score(y_train, ymodel_three_train), '\n')

# Model three test data
print("Model three test data:")
print("Mean Squared Error: ", mean_squared_error(y_test, ymodel_three_test))
print("Mean Absolute Error: ", mean_absolute_error(y_test, ymodel_three_test))
print("R2 Score: ", r2_score(y_test, ymodel_three_test), '\n')

# Additional information
print("Iteration count:")
print("Model one: ", model_one.n_iter_)
print("Model two: ", model_two.n_iter_)
print("Model three: ", model_three.n_iter_, '\n')

print("Layer count including input and output layers:")
print("Model one: ", model_one.n_layers_)
print("Model two: ", model_two.n_layers_)
print("Model three: ", model_three.n_layers_)
# %%