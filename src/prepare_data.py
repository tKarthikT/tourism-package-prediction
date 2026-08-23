import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = 'data/Tourism.csv'
ARTIFACT_DIR = 'artifacts'

def prepare_data():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=['Unnamed: 0', 'CustomerID'], errors='ignore')
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace({'Fe Male': 'Female', 'Male ': 'Male'})
    df = df.drop_duplicates().copy()
    X = df.drop(columns='ProdTaken')
    y = df['ProdTaken']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    X_train.to_csv(f'{ARTIFACT_DIR}/X_train.csv', index=False)
    X_test.to_csv(f'{ARTIFACT_DIR}/X_test.csv', index=False)
    y_train.to_csv(f'{ARTIFACT_DIR}/y_train.csv', index=False)
    y_test.to_csv(f'{ARTIFACT_DIR}/y_test.csv', index=False)
    print('Data preparation complete')
    print(f'Training shape: {X_train.shape}')
    print(f'Testing shape: {X_test.shape}')

if __name__ == '__main__':
    prepare_data()
