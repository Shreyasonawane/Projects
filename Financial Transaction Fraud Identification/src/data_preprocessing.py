import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(path):
    df = pd.read_csv(path)
    df['Amount'] = StandardScaler().fit_transform(df[['Amount']])
    df = df.drop(['Time'], axis=1)
    return df
