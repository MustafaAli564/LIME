import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def get_data():

    df1 = pd.read_csv('../../Semester-8/Probattle/train_set.csv')
    df2 = pd.read_csv('../../Semester-8/Probattle/test_set.csv')

    df1 = pd.get_dummies(df1, columns=['x84'], prefix='x84')
    df2 = pd.get_dummies(df2, columns=['x84'], prefix='x84')

    df1 = pd.get_dummies(df1, columns=['x85'], prefix='x85')
    df2 = pd.get_dummies(df2, columns=['x85'], prefix='x85')

    df1 = df1.drop(columns=['row_id'])
    df2 = df2.drop(columns=['row_id'])

    range_mapping = {
        '18-25': 1,
        '26-40': 2,
        '41-60': 3,
        '60+': 4
    }

    # Apply the mapping
    df1['x86'] = df1['x86'].map(range_mapping)
    df2['x86'] = df2['x86'].map(range_mapping)

    # Handle missing values (e.g., replace NaN with the median)
    df1['x86'].fillna(df1['x86'].median(), inplace=True)
    df2['x86'].fillna(df2['x86'].median(), inplace=True)

    # SIMPLE IMPUTER
    imputer = SimpleImputer(strategy='mean')
    df1_imputed = pd.DataFrame(imputer.fit_transform(df1), columns=df1.columns)
    df1_imputed.isnull().sum()
    df2_imputed = pd.DataFrame(imputer.fit_transform(df2), columns=df2.columns)
    df2_imputed.isnull().sum()

    # SPLITTING THE TARGET VARIABLE
    x = df1_imputed.drop(columns=['target'])
    y = df1_imputed['target']

    # MINMAX SCALING
    scale = MinMaxScaler()
    x = scale.fit_transform(x)
    df2_imputed = scale.fit_transform(df2_imputed)

    return x,y