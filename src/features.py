"""
Feature engineering library.
Each function takes a DataFrame and returns a new one with added columns.
Never modify existing functions — add versioned ones (v2, v3, ...).
"""
import pandas as pd


def interactions_v1(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise interactions between the three strongest EDA predictors."""
    df = df.copy()
    df['Glucose_BMI'] = df['Glucose'] * df['BMI']
    df['Glucose_Age'] = df['Glucose'] * df['Age']
    df['BMI_Age']     = df['BMI']     * df['Age']
    return df
