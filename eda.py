import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('data/data.csv')

# Basic EDA
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())

# Dropping columns that are not useful for prediction
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)
if 'Unnamed: 32' in df.columns:
    df.drop('Unnamed: 32', axis = 1, inplace = True)

# Removing Duplicates
df.drop_duplicates(inplace=True)

# Encoding the target variable
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])
print(df.corr()['diagnosis'].sort_values(ascending=False))

# Saving the data to data folder
if not os.path.exists('data'):
    os.makedirs('data')

df.to_csv('data/cleaned_data.csv', index = False)
print("Cleaned data saved to data/cleaned_data.csv")