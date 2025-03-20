#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Lung Cancer Predictor Model Training Script

This script loads the lung cancer dataset, performs feature engineering,
splits the data, trains a Logistic Regression model using a Pipeline with hyperparameter tuning via GridSearchCV,
evaluates the model, and saves the best model pipeline to a file for use in prediction.
"""

import logging
import os
from typing import Tuple, Dict

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from joblib import dump

# Configure logging for traceability
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(filepath):
        logging.error("File not found: %s", filepath)
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath)
    logging.info("Data loaded successfully with shape: %s", df.shape)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the DataFrame.
    Converts the target variable to numeric and creates additional features.

    Args:
        df (pd.DataFrame): Original DataFrame.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    # Convert target variable to numeric (0 for NO, 1 for YES)
    df['PULMONARY_DISEASE'] = df['PULMONARY_DISEASE'].map({'NO': 0, 'YES': 1})
    
    # Create interaction and non-linear features
    df['SMOKING_AGE_Interaction'] = df['SMOKING'] * df['AGE']
    df['SMOKING_FAMILY_HISTORY_Interaction'] = df['SMOKING'] * df['FAMILY_HISTORY']
    df['AGE_Squared'] = df['AGE'] ** 2
    df['Respiratory_Issue_Score'] = df['BREATHING_ISSUE'] + df['CHEST_TIGHTNESS'] + df['THROAT_DISCOMFORT']
    
    logging.info("Feature engineering completed.")
    return df


def split_data(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split the dataset into training, testing, and validation sets.

    Args:
        df (pd.DataFrame): The complete DataFrame.
        target (str): The target variable column name.

    Returns:
        Tuple containing training, testing, and validation splits.
    """
    X = df.drop(columns=[target])
    y = df[target]
    # 70% training, 30% temporary set
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    # Split temporary set equally into test and validation (15% each)
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    logging.info("Data split into training (70%%), testing (15%%), and validation (15%%).")
    return X_train, X_test, X_val, y_train, y_test, y_val


def build_pipeline() -> Pipeline:
    """
    Build a scikit-learn pipeline with StandardScaler and LogisticRegression.

    Returns:
        Pipeline: The machine learning pipeline.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=1000))
    ])
    return pipeline


def tune_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Pipeline, Dict]:
    """
    Perform hyperparameter tuning using GridSearchCV.

    Args:
        pipeline (Pipeline): The machine learning pipeline.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        Tuple: Best model pipeline and best hyperparameters.
    """
    param_grid = {
        'logreg__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'logreg__penalty': ['l1', 'l2'],
        'logreg__solver': ['liblinear']
    }
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    logging.info("Grid Search best parameters: %s", grid.best_params_)
    logging.info("Best CV Accuracy: %.4f", grid.best_score_)
    return grid.best_estimator_, grid.best_params_


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series, dataset_name: str) -> None:
    """
    Evaluate the trained model and log performance metrics.

    Args:
        model (Pipeline): Trained model pipeline.
        X (pd.DataFrame): Features for evaluation.
        y (pd.Series): True target values.
        dataset_name (str): Name of the dataset (e.g., "Test", "Validation").
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)

    logging.info("%s Metrics:", dataset_name)
    logging.info("Accuracy: %.4f", acc)
    logging.info("Precision: %.4f", prec)
    logging.info("Recall: %.4f", rec)
    logging.info("F1 Score: %.4f", f1)
    logging.info("AUC-ROC: %.4f", auc)
    logging.info("Classification Report:\n%s", classification_report(y, y_pred))


def save_model(model: Pipeline, filepath: str) -> None:
    """
    Save the trained model pipeline to disk.

    Args:
        model (Pipeline): The trained model pipeline.
        filepath (str): Destination path for saving the model.
    """
    dump(model, filepath)
    logging.info("Model pipeline saved to %s", filepath)


def main():
    # Define file path for dataset
    file_path = 'Lung Cancer Dataset.csv'
    
    # Load the dataset
    df = load_data(file_path)
    
    # Perform feature engineering
    df = feature_engineering(df)
    
    # Split the dataset into training, testing, and validation sets
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(df, target='PULMONARY_DISEASE')
    
    # Build the machine learning pipeline
    pipeline = build_pipeline()
    
    # Tune hyperparameters using GridSearchCV
    best_pipeline, best_params = tune_model(pipeline, X_train, y_train)
    
    # Evaluate the model on validation and test sets (pipeline handles scaling internally)
    evaluate_model(best_pipeline, X_val, y_val, dataset_name="Validation")
    evaluate_model(best_pipeline, X_test, y_test, dataset_name="Test")
    
    # Save the best model pipeline to a file for use in prediction
    model_save_path = "best_lung_cancer_model.joblib"
    save_model(best_pipeline, model_save_path)


if __name__ == "__main__":
    main()
