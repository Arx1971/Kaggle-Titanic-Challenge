import pandas as pd
from sklearn.tree import DecisionTreeRegressor

home_data = pd.read_csv('train.csv')

Y = home_data['SalePrice']  # Prediction Target

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr',
            'TotRmsAbvGrd']  # Extracting the features from original given data

X = home_data[features]
