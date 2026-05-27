import pandas as pd
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('data.csv')

print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())

if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

print(df.corr()['diagnosis'].sort_values(ascending=False))