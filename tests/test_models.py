import pandas as pd
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from models_rf import train_random_forest
from models_dt import train_decision_tree

@pytest.fixture
def mock_preprocessor_simple():
    return ColumnTransformer(
        transformers=[('pass', 'passthrough', [0])]
    )

@pytest.fixture
def mock_train_data():
    X_train = pd.DataFrame({'feature': np.arange(15)})
    y_train = np.array([0, 1] * 7 + [1]) 
    return X_train, y_train

#Tests du Random Forest

def test_random_forest_pipeline_structure(mock_preprocessor_simple, mock_train_data):
    X_train, y_train = mock_train_data
    rf_model = train_random_forest(mock_preprocessor_simple, X_train, y_train)
    
    assert isinstance(rf_model, Pipeline)
    
    classifier_step = rf_model.steps[1][1]
    assert isinstance(classifier_step, RandomForestClassifier)


#Tests du Decision Tree

def test_decision_tree_pipeline_structure(mock_preprocessor_simple, mock_train_data):
    X_train, y_train = mock_train_data
    dt_model = train_decision_tree(mock_preprocessor_simple, X_train, y_train)
    
    assert isinstance(dt_model, Pipeline)
    
    classifier_step = dt_model.steps[1][1]
    assert isinstance(classifier_step, DecisionTreeClassifier)