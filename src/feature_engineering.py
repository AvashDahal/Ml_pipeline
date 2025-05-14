import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml
import sys

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def main():
    try:
        # Print debugging information
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for params.yaml in current directory: {os.path.exists('params.yaml')}")
        print(f"Looking for params.yaml in parent directory: {os.path.exists('../params.yaml')}")

        # Hardcode parameters instead of loading from file
        max_features = 50
        logger.info(f"Using hardcoded max_features={max_features}")
        print(f"Using hardcoded max_features={max_features}")

        # Make sure the required directories exist
        os.makedirs('./data/interim', exist_ok=True)
        os.makedirs('./data/processed', exist_ok=True)
        print("Created required directories")

        # Load data
        try:
            train_data = pd.read_csv('./data/interim/train_processed.csv')
            test_data = pd.read_csv('./data/interim/test_processed.csv')
            print("Successfully loaded training and test data")
        except Exception as e:
            print(f"Error loading data: {e}")
            logger.error(f"Error loading data: {e}")

            # Check if files exist
            print(f"Train file exists: {os.path.exists('./data/interim/train_processed.csv')}")
            print(f"Test file exists: {os.path.exists('./data/interim/test_processed.csv')}")

            # Try alternative paths
            alt_train_path = '../data/interim/train_processed.csv'
            alt_test_path = '../data/interim/test_processed.csv'
            print(f"Alternate train file exists: {os.path.exists(alt_train_path)}")
            print(f"Alternate test file exists: {os.path.exists(alt_test_path)}")

            # If alternate paths exist, use them
            if os.path.exists(alt_train_path) and os.path.exists(alt_test_path):
                train_data = pd.read_csv(alt_train_path)
                test_data = pd.read_csv(alt_test_path)
                print("Successfully loaded training and test data from alternate paths")
            else:
                raise

        # Fill NA values
        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)

        # Apply TF-IDF
        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        print("TF-IDF transformation completed")

        # Save processed data
        train_output_path = './data/processed/train_tfidf.csv'
        test_output_path = './data/processed/test_tfidf.csv'

        train_df.to_csv(train_output_path, index=False)
        test_df.to_csv(test_output_path, index=False)

        print(f"Data saved to {train_output_path} and {test_output_path}")
        print(f"Output files exist: train={os.path.exists(train_output_path)}, test={os.path.exists(test_output_path)}")

        logger.info('Feature engineering process completed successfully')
        print("Feature engineering completed successfully!")

    except Exception as e:
        logger.error(f'Failed to complete the feature engineering process: {e}')
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()