import pandas as pd
from pathlib import Path

EXPECTED_COLUMNS = [
    'ProdTaken', 'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch',
    'Occupation', 'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups',
    'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus', 'NumberOfTrips',
    'Passport', 'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'
]

DATA_PATH = Path('data/Tourism.csv')

def validate_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'Dataset not found: {DATA_PATH}')
    df = pd.read_csv(DATA_PATH)
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f'Missing expected columns: {missing_cols}')
    print('Dataset validation successful')
    print(f'Shape: {df.shape}')
    print(df.dtypes)
    print(df.isnull().sum())
    print(df['ProdTaken'].value_counts(normalize=True))

if __name__ == '__main__':
    validate_data()
