import pytest
import pandas as pd
import os
import numpy as np
from ml.data import process_data
from ml.model import train_model, inference

@pytest.fixture(scope="session")
def data():
    """
    Fixture to load the census data for testing.
    """
    data_path = os.path.join("data", "census.csv")
    return pd.read_csv(data_path)

@pytest.fixture
def cat_features():
    return [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]

def test_process_data_shape(data, cat_features):
    """
    Test that the process_data function returns the expected 
    feature and label shapes.
    """
    X, y, encoder, lb = process_data(
        data, 
        categorical_features=cat_features, 
        label="salary", 
        training=True
    )
    
    assert len(X) == len(y)
    assert X.shape[1] > len(cat_features)  # One-hot encoding should expand columns
    assert len(np.unique(y)) == 2          # Binary classification

def test_train_model_type(data, cat_features):
    """
    Test that the train_model function returns a valid model 
    object with a predict method.
    """
    X, y, _, _ = process_data(
        data.iloc[:100], # Use a subset for speed
        categorical_features=cat_features, 
        label="salary", 
        training=True
    )
    model = train_model(X, y)
    
    assert hasattr(model, "predict")

def test_inference_output(data, cat_features):
    """
    Test that the inference function returns predictions 
    of the correct type and length.
    """
    # Setup
    X, y, encoder, lb = process_data(
        data.iloc[:50], 
        categorical_features=cat_features, 
        label="salary", 
        training=True
    )
    model = train_model(X, y)
    
    # Run inference
    preds = inference(model, X)
    
    assert len(preds) == len(X)
    assert isinstance(preds, np.ndarray)
    # Check that predictions are binary (0 or 1)
    assert set(preds).issubset({0, 1})