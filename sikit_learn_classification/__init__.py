import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

home_data = pd.read_csv('train.csv')

Y = home_data['SalePrice']  # Prediction Target

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr',
            'TotRmsAbvGrd']  # Extracting the features from original given data

X = home_data[features]

train_X, val_X, train_Y, val_Y = train_test_split(X, Y, random_state=0)

data_model = DecisionTreeRegressor()
data_model.fit(train_X, train_Y)
val_predictions = data_model.predict(val_X)

