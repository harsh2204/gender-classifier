import pandas as pd
from pprint import pprint
from sklearn.naive_bayes import GaussianNB # Naive Bayes

# Dataset from - https://www.kaggle.com/hb20007/gender-classification

gnb = GaussianNB()

df = pd.read_csv("Transformed Data Set - Sheet1.csv")
df = df.to_dict('split')
dataset = []
genders = []
# pprint(df)

for data in df['data']:
    genders.append(data.pop())
    dataset.append(data)

# pprint(genders)
# pprint(dataset)
print("Dataset Loaded!")

gnb.fit(dataset,genders)
# Doesn't work :(
test = ['Cool','Pop','Beer','Coca Cola/Pepsi']

prediction = gnb.predict(test)
print(f'Prediction for [{test}]: {prediction}')