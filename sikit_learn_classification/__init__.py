import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_x, train_y)
    preds_val = model.predict(val_x)
    mae = mean_absolute_error(val_y, preds_val)
    return mae


home_data = pd.read_csv('train.csv')

Y = home_data['SalePrice']  # Prediction Target

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr',
            'TotRmsAbvGrd']  # Extracting the features from original given data

X = home_data[features]

train_X, val_X, train_Y, val_Y = train_test_split(X, Y, random_state=0)

tree_sizes = [5, 25, 50, 100, 250, 500]
min_val = get_mae(tree_sizes[0], train_X, val_X, train_Y, val_Y)
index = 0
print(min_val)
for i in range(1, len(tree_sizes)):
    temp = get_mae(tree_sizes[i], train_X, val_X, train_Y, val_Y)
    print(temp)
    if temp < min_val:
        min_val = temp
        index = i

final_model = DecisionTreeRegressor(max_leaf_nodes=tree_sizes[index], random_state=0)
final_model.fit(train_X, train_Y)
val_predictions = final_model.predict(val_X)
