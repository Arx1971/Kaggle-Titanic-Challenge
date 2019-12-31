import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
Y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = DecisionTreeClassifier(max_leaf_nodes=15, random_state=1)
model.fit(X, Y)
predictions = model.predict(X_test)
print(predictions)
print(model.score(X_test, predictions))

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
