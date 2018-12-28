import pandas as pd
from pprint import pprint
from sklearn.naive_bayes import GaussianNB # Naive Bayes

# Dataset from - https://www.kaggle.com/hb20007/gender-classification

gnb = GaussianNB()

df = pd.read_csv("Transformed Data Set - Sheet1.csv")
dict_df = df.to_dict('split')
dataset = []
genders = []
# pprint(df)

sets = []
for head in df:
    sets += list(set(df[head]))

# print(sets)

def mapper(x): # Ew
    return sets.index(x)   

for data in dict_df['data']:
    genders.append(data.pop())
    dataset.append(list(map(mapper, data))) 
    # Don't know enough about ML currently to implement this in a better way
    # But it works atleast

pprint(dataset)
print("Dataset Loaded!")

gnb.fit(dataset,genders)
test = ['Cool','Pop','Beer','Coca Cola/Pepsi']

array = list(map(mapper, test))
prediction = gnb.predict([array])
print(f'Prediction for [{test}]: {prediction}')