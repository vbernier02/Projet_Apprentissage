import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']
    return X, y


def encode_target(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le.classes_


def build_preprocessor(X):
    numerical_features = X.select_dtypes(include=np.number).columns
    categorical_features = X.select_dtypes(include='object').columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    return preprocessor


def split_data(X, y_encoded):
    return train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
