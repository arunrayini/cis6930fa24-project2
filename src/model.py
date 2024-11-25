import os
import nltk
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from src.feature_extraction import extract_features
from src.predict import predict
from src.utils import load_data

nltk.download('punkt')

def create_directory_structure():
    """
    Ensuring that the necessary directories exist.
    """
    directories = ["data", "output"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def create_sample_data(file_path):
    """
    Creating a sample training data file if needed.
    """
    sample_data = [
        ['training', 'Jack Black', 'I have always liked █████ █████ so you know what you will get out of him.'],
        ['training', 'Jack Bender', 'If █████ █████ continues with this episode from the season premiere, he...'],
        ['validation', 'JJ Abrams', 'Mr █████████, what did you think to create this amazing story?']
    ]
    
    # Creating DataFrame with explicit column names
    df = pd.DataFrame(sample_data, columns=['split', 'name', 'context'])
    
    # Saving with tab separator and UTF-8 encoding
    df.to_csv(file_path, sep='\t', index=False, encoding='utf-8')
    print(f"Created sample data file at {file_path}")

def train_model(training_file):
    """
    Train the model using the training data.
    """
    print(f"\nTraining model using file: {training_file}")

    try:
        # Loading training data
        train_data = load_data(training_file, expected_columns=['split', 'name', 'context'])

        # Spliting data into training and validation sets
        training_set = train_data[train_data['split'] == 'training']
        validation_set = train_data[train_data['split'] == 'validation']

        if training_set.empty:
            raise ValueError("Training data is empty.")

        # Extracting features and vectorize
        train_features = [extract_features(row['context']) for _, row in training_set.iterrows()]
        train_labels = training_set['name'].values

        vec = DictVectorizer()
        X_train = vec.fit_transform(train_features)

        # Training RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, train_labels)
        print("Model training complete.")

        # Validating the model
        if not validation_set.empty:
            print("Validating model...")
            val_features = [extract_features(row['context']) for _, row in validation_set.iterrows()]
            X_val = vec.transform(val_features)
            y_val = validation_set['name'].values
            y_pred = model.predict(X_val)

            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

            print("\nValidation Metrics:")
            print(f"Accuracy: {accuracy:.2f}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1-Score: {f1:.2f}")

        return model, vec
    except Exception as e:
        print(f"Error training model: {e}")
        return None, None

def process_test_file(test_file, model, vec, submission_file):
    """
    Processing the test file and generate a submission file with predictions.
    """
    print(f"\nProcessing test file: {test_file}")

    try:
        # Load test data with is_test=True flag
        test_data = load_data(test_file, expected_columns=['id', 'context'], is_test=True)

        predictions = []
        for _, row in test_data.iterrows():
            context = row['context']
            predicted_name = predict(model, vec, context)
            predictions.append({
                'id': row['id'],
                'name': predicted_name  # Using 'name' instead of 'predicted_name' for submission format
            })

        # Saving predictions to submission file
        submission_df = pd.DataFrame(predictions)
        submission_df.to_csv(submission_file, sep='\t', index=False)
        print(f"Submission file created: {submission_file}")
        #print(submission_df.head())
        
    except Exception as e:
        print(f"Error processing test file: {e}")
        raise

# model.py - updating the main function:
def main():
    """
    Main function to execute the unredactor pipeline.
    """
    try:
        create_directory_structure()
        data_path = "data/source_unredactor.tsv"
        test_path = "data/test.tsv"
        submission_path = "data/submission.tsv"

        # Training the model
        model, vec = train_model(data_path)
        if model and vec:
            # Processing the test file and generate the submission
            process_test_file(test_path, model, vec, submission_path)
            print("\nComplete! Check submission.tsv for results.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("\nPlease ensure your files meet these requirements:")
        print("1. Training file (source_unredactor.tsv) should have columns: split, name, context")
        print("2. Test file (test.tsv) should have columns: id, context")
        print("3. All files should be tab-separated TSV files")

def main():
    """
    Main function to execute the unredactor pipeline.
    """
    try:
        create_directory_structure()
        data_path = "data/source_unredactor.tsv"
        test_path = "data/test.tsv"
        submission_path = "data/submission.tsv"

        # Ensuring training data exists and is valid
        if not os.path.exists(data_path):
            print(f"{data_path} not found. Generating sample training data...")
            create_sample_data(data_path)

        # Training the model
        model, vec = train_model(data_path)
        if model and vec:
            # Processing the test file and generate the submission
            process_test_file(test_path, model, vec, submission_path)
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
