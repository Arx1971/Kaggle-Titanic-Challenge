import pandas as pd

train_data = pd.DataFrame(pd.read_csv('train.csv'))

total_survivor = train_data[train_data.Survived == 1]
female_survivor = total_survivor[total_survivor.Sex == 'female']
male_survivor = total_survivor[total_survivor.Sex == 'male']

print(male_survivor)
