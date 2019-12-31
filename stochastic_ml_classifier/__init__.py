import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
Y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
model = RandomForestClassifier(n_estimators=500, random_state=1)
model.fit(X, Y)
predictions = model.predict(X_test)
accuracy = model.score(X_test, predictions)
print(accuracy)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
