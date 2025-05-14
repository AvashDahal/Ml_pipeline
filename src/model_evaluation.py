# Use this alternative file if you continue to have issues with dvclive
import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import yaml
import sys
import json

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise


def save_dvc_metrics_manually(metrics: dict, params: dict) -> None:
    """Save DVC-compatible metrics without using dvclive."""
    try:
        # Create the directory structure that DVC expects
        dvc_dir = "dvc_manual_metrics"
        os.makedirs(dvc_dir, exist_ok=True)

        # Save metrics to a format DVC can consume
        metrics_file = os.path.join(dvc_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.debug(f"Metrics saved to {metrics_file}")

        # Save parameters to a format DVC can consume
        params_file = os.path.join(dvc_dir, "params.json")
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=4)
        logger.debug(f"Parameters saved to {params_file}")

        # Create a DVC-compatible summary file
        summary_data = {
            "metrics": metrics,
            "params": params
        }
        summary_file = os.path.join(dvc_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=4)
        logger.debug(f"Summary saved to {summary_file}")

        logger.info("DVC-compatible metrics and parameters saved manually")
    except Exception as e:
        logger.error(f"Error saving DVC metrics manually: {e}")
        logger.warning("Continuing despite manual DVC metrics error")


def main():
    try:
        # Create output directories first to avoid permission issues
        os.makedirs('reports', exist_ok=True)

        params = load_params(params_path='../params.yaml')
        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, X_test, y_test)

        # Save metrics to the reports directory
        save_metrics(metrics, 'reports/metrics.json')
        logger.debug('Metrics saved successfully')

        # Save DVC-compatible metrics manually (no dvclive dependency)
        save_dvc_metrics_manually(metrics, params)

        logger.debug('Model evaluation process completed successfully')
        print("Model evaluation completed successfully!")
        print(f"Metrics: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, "
              f"Recall={metrics['recall']:.4f}, AUC={metrics['auc']:.4f}")

    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()