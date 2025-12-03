import pandas as pd
import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from preprocessing import load_data, encode_target, build_preprocessor, split_data

DATA_PATH = "data/raw/ObesityDataSet_raw_and_data_sinthetic.csv"

@pytest.fixture
def mock_data():
    return pd.DataFrame({
        'NObeyesdad': ['C1', 'C2', 'C1', 'C2', 'C1', 'C2', 'C1', 'C2', 'C1', 'C2'],
        'Age': [30, 40, 50, 60, 20, 30, 40, 50, 60, 70]
    })


def test_data_loading_real_file():
    X, y = load_data(DATA_PATH)

    assert X.shape[0] > 0
    assert X.shape[1] > 1
    assert 'NObeyesdad' not in X.columns
    assert y.name == 'NObeyesdad'


def test_target_encoding_type(mock_data):
    y = mock_data['NObeyesdad']
    y_encoded, target_names = encode_target(y)
    
    #Vérifie que la sortie est un tableau d'entiers
    assert np.issubdtype(y_encoded.dtype, np.integer)
    #Vérifie que toutes les classes ont été encodées
    assert len(target_names) == 2


def test_preprocessor_is_columntransformer(mock_data):
    #Vérifie que build_preprocessor retourne un objet ColumnTransformer
    X = mock_data.drop('NObeyesdad', axis=1)
    preprocessor = build_preprocessor(X)
    
    assert isinstance(preprocessor, ColumnTransformer)
    assert len(preprocessor.transformers) == 2


def test_split_data_integrity(mock_data):
    #Vérifie que le split maintient le nombre total d'échantillons
    X = mock_data.drop('NObeyesdad', axis=1)
    y = mock_data['NObeyesdad']
    y_encoded, _ = encode_target(y)
    
    X_train, X_test, y_train, y_test = split_data(X, y_encoded)
    
    assert len(X_train) + len(X_test) == len(X)
    assert len(X_test) == 2