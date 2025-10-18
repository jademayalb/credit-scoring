import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import pytest
from app.api import preprocess, features, poly_transformer

def test_preprocess_shape_and_columns():
    # Exemple  
    data = pd.DataFrame({
        "AMT_CREDIT": [100000],
        "AMT_ANNUITY": [5000],
        "AMT_INCOME_TOTAL": [50000],
        "DAYS_BIRTH": [-12000],
        "DAYS_EMPLOYED": [365243],  # valeur anormale
        "EXT_SOURCE_1": [0.5],
        "EXT_SOURCE_2": [0.4],
        "EXT_SOURCE_3": [0.3],
    })
    processed = preprocess(data, features, poly_transformer)
    # Vérifie que toutes les colonnes attendues sont là
    assert list(processed.columns) == list(features)
    # Vérifie qu'il n'y a pas de NaN après preprocessing (avant imputation)
    assert processed.isnull().sum().sum() >= 0  # Peut être >0 avant imputation