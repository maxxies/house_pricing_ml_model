import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

#--> Load data set
data_path = '../../Data/housing.csv'
housing_data = pd.read_csv(data_path)

#--> Cleaning dataset
# Combining attributes
housing_data['population_per_household'] = housing_data['population'] / housing_data['households']
housing_data['rooms_per_household'] = housing_data['total_rooms'] / housing_data['households']
housing_data['bedrooms_per_household'] = housing_data['total_bedrooms'] / housing_data['households']
housing_data['median_income_per_household'] = housing_data['median_income'] / housing_data['households']

# Correlation matrix
corr_matrix = housing_data.corr()
# print(corr_matrix['median_house_value'].sort_values(ascending = False))

# Fill in missing spaces
imputer = SimpleImputer(strategy='median')       # Sets method for what values to replace missing values
housing_data_num = housing_data.drop("ocean_proximity", axis=1)  # Removes non numerical columns
imputer.fit(housing_data_num)
new_data_num = imputer.transform(housing_data_num)
new_housing_data_num = pd.DataFrame(new_data_num, columns=housing_data_num.columns)
# print(new_housing_data_num.info())

# Replacing text with numbers
proximity = housing_data[['ocean_proximity']]
cat_encoder = OrdinalEncoder()
housing_proximity_cat = cat_encoder.fit_transform(proximity)
new_housing_data_num['ocean_proximity'] = housing_proximity_cat  # Adding new numerical ocean proximity to dataframe


#--> Display data
housing_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing_data["population"]/100, label="population", figsize=(10,7),
                  c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
# plt.show()


#--> Splitting dataframe
housing_data = new_housing_data_num.drop('median_house_value', axis=1)  # Removes house prices from data
housing_labels = new_housing_data_num['median_house_value']    # Sets house labels in a data frame

train_X, test_X, train_y, test_y = train_test_split(housing_data, housing_labels, test_size=0.15, random_state=47, shuffle=True)
# Getting the best tree size
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# find the ideal tree size from candidate_max_leaf_nodes
scores = {leaf_size: get_mae(leaf_size, train_X, test_X, train_y, test_y) for leaf_size in max_leaf_nodes}
# Store the best value of max_leaf_nodes
best_tree_size = min(scores, key=scores.get)


#--> Training model
model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size)
# model = LinearRegression()
# Fitting data
model.fit(train_X, train_y)
# Making predictions
predictions = model.predict(test_X)


#--> Validating model
scores = cross_val_score(model, train_X, train_y, scoring="neg_mean_squared_error", cv=10)   #Checks for scores in ssets of training data
model_rmse_scores = np.sqrt(-scores)      # Calculates the performance metric using root mean square method
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# Accuracy check
print("Accuracy : {}".format(model.score(test_X, test_y)))
display_scores(model_rmse_scores)





