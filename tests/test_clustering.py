import pandas as pd
import numpy as np
import pytest
from clustering import run_kmeans, run_cah


@pytest.fixture
def mock_processed_data():
    N_SAMPLES = 10
    N_FEATURES = 3
    K_CLUSTERS = 2

    X_processed = np.random.rand(N_SAMPLES, N_FEATURES)
    X_processed_df = pd.DataFrame(X_processed, columns=[f'f{i}' for i in range(N_FEATURES)])
    y_encoded = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    return X_processed_df, y_encoded, K_CLUSTERS


#Tests du K-means

def test_kmeans_output_types(mock_processed_data):

    X_processed_df, y_encoded, K = mock_processed_data
    
    clusters, silhouette, ari, contingency = run_kmeans(X_processed_df, y_encoded, K)
    
    #Vérification des types
    assert isinstance(clusters, np.ndarray)
    assert isinstance(silhouette, float)
    assert isinstance(ari, float)
    assert isinstance(contingency, pd.DataFrame)
    
    #Vérification des dimensions
    assert len(clusters) == len(y_encoded)
    assert contingency.shape == (K, K)

#Tests du CAH

def test_cah_output_types(mock_processed_data):

    X_processed_df, y_encoded, K = mock_processed_data
    clusters, silhouette, ari, contingency = run_cah(X_processed_df, y_encoded, K)
    
    assert isinstance(clusters, np.ndarray)
    assert isinstance(silhouette, float)
    assert isinstance(ari, float)
    assert isinstance(contingency, pd.DataFrame)

    assert len(clusters) == len(y_encoded)
    assert contingency.shape == (K, K)