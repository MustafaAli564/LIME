import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical

df = pd.read_csv("../Datasets/KDDCup99.csv")

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'])

df.to_csv('../Datasets/dataset.csv', index=False)