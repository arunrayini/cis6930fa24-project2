# cis6930fa24 -- Project 2

Name: Arun Kumar Reddy Rayini

# Project Description:
This project aims to create a robust and scalable machine learning pipeline capable of predicting redacted names from textual data. In various real-world applications, such as anonymized datasets, names or other identifying information are often replaced with redacted symbols, represented here by █ characters. The goal of this pipeline is to reconstruct or predict the redacted names using the surrounding context within the sentences.

# Project Objectives
Model Development: Leverage machine learning techniques, specifically a Random Forest Classifier, to predict redacted names based on context.
Feature Extraction: Implement efficient feature extraction methods to analyze and utilize the surrounding textual data of the redacted portions, including words before and after the redaction, sentence structure, and additional patterns.
Pipeline Design: Develop a clear, modular, and reusable pipeline for training, validating, and testing the model to ensure scalability and adaptability for future datasets or use cases.
Ease of Use: Provide a streamlined and user-friendly framework, accompanied by comprehensive documentation, allowing users with minimal technical expertise to understand, replicate, and deploy the system.

---

# Key Features
Contextual Understanding: The system extracts and processes contextual information to predict the redacted names, leveraging linguistic insights such as surrounding words and redaction patterns.
Modular Architecture: The project is structured with distinct components for feature extraction, model training, data handling, and prediction, ensuring flexibility and ease of maintenance.
Intermediate Validation: During training, the pipeline generates intermediate results (unredactor.tsv and unredactor_cleaned.tsv) to assess model performance, providing insights into accuracy and limitations.
Final Predictions: The system processes an unseen test dataset (test.tsv) and generates a submission.tsv file containing the predicted names for each redacted sentence.

The system processes two input files:
•	source_unredactor.tsv: A dataset used for training and validation.
•	test.tsv: A dataset for testing the model with unknown names.

It produces three outputs:
•	unredactor.tsv: Intermediate validation.
•	unredactor_cleaned.tsv: A filtered version of unredactor.tsv.
•	submission.tsv: Final predictions for the test dataset.
The pipeline is designed to be modular, with separate functions handling feature extraction, data loading, model training, prediction, and evaluation.

---

# Directory Structure

# Data Directory:
source_unredactor.tsv: The primary dataset containing labeled training and validation data with columns for split, name, and context.
test.tsv: Contains unlabeled sentences where redacted names need to be predicted, with columns id and context.
unredactor.tsv: Stores intermediate validation results for evaluating model performance.
unredactor_cleaned.tsv: A cleaned version of unredactor.tsv, predictions for a focused evaluation.
submission.tsv: Final output file containing predictions for the test.tsv dataset, formatted with id and predicted name.

---

# Source Code Directory (src)

feature_extraction.py: Defines functions to process sentences and extract features for machine learning.
model.py: Implements the end-to-end pipeline, including training the model, validating it on intermediate data, and predicting results for the test set.
predict.py: Houses the logic to generate predictions using the trained Random Forest model and the extracted contextual features.
utils.py: Provides helper functions for data loading, preprocessing, and validation to ensure consistency and correctness.

Submission file is in Data Directory:
Submission Files: Contains the generated submission.tsv file with final predictions.
Intermediate Results: Includes intermediate outputs like unredactor.tsv and unredactor_cleaned.tsv, enabling step-by-step validation of the pipeline.

---

# Setup Instructions
Install Python: Ensure Python 3.7 or later is installed.

Install Dependencies:
pip install nltk pandas scikit-learn

Prepare the Input Files:
Place source_unredactor.tsv and test.tsv in the data directory.

---

# How to Run
Ensure Directory Structure:

Verify that the data  directory exist.

  Run the Pipeline:   python -m src.model


Outputs:

Check the data directory for generated files:
submission.tsv: Final predictions for the test dataset.
unredactor.tsv and unredactor_cleaned.tsv: Intermediate validation results.

---

# Code Explanation

# Feature Extraction (feature_extraction.py)
Purpose:
Extracts meaningful contextual features from sentences containing redacted portions to help the model predict the missing names.

Detailed Explanation

nltk.download('punkt'):
This downloads the Punkt tokenizer from the Natural Language Toolkit (NLTK). The Punkt tokenizer splits sentences into words, ensuring accurate tokenization for natural language processing.
Function: extract_features(context)

Input: A sentence that includes redacted portions represented by █ characters.
Process:
Tokenization:
The nltk.word_tokenize function breaks the input sentence into individual words, preserving the sequence and context of the words.

Identify Redacted Index:
The context.find("█") function searches for the redacted characters in the sentence and returns their position.
Feature Extraction:
previous_word: Identifies the word immediately preceding the redacted portion. 
next_word: Identifies the word immediately following the redacted portion. 
length_redacted: Counts the total number of █ characters, indicating the length of the redacted name.
context: Cleans the context by replacing all █ characters, providing a cleaner input for feature processing.
Output:
Returns a dictionary of extracted features.

---

# Model Training and Testing (model.py)
Purpose:
This script orchestrates the entire machine learning pipeline, from training the model to generating predictions on the test data.

Key Functions and Steps:

create_directory_structure()

Purpose: Ensures the existence of necessary directories (data and output).
Process:
Checks if the data and output directories exist.
If not, it creates these directories to ensure the pipeline functions without errors.
Output: Log messages confirming directory creation if they were missing.

create_sample_data(file_path)

Purpose: Creates a sample source_unredactor.tsv file for training and validation.
Process:
Generates synthetic data with split (training or validation), name, and context.
Saves this data as a tab-separated file at the specified path.
Use Case: Helps users test the pipeline even without real data.

train_model(training_file)

Purpose: Trains the Random Forest Classifier on the training data and validates its performance.
Steps:
Data Loading:
Uses utils.load_data() to load the source_unredactor.tsv file and validate its structure.
Splits the data into training_set and validation_set based on the split column.

Feature Extraction:
Extracts contextual features for each sentence using feature_extraction.extract_features.
Example Features: Previous word, next word, redacted length.
Vectorization:
Converts the extracted features into a machine-readable format using DictVectorizer.
Vectorized features allow the machine learning model to process contextual information numerically.

Model Training:
Uses a RandomForestClassifier from scikit-learn, a robust algorithm for classification tasks.
Trains the model on the vectorized features and their corresponding labels (name).

Validation:
Evaluates the model on the validation set. Calculates metrics like:

Accuracy: Percentage of correct predictions.
Precision: Ratio of true positive predictions to all positive predictions.
Recall: Ratio of true positives to all actual positives.
F1-Score: Harmonic mean of precision and recall.

process_test_file(test_file, model, vec, submission_file)

Purpose: Generates predictions for the test.tsv file and saves them in submission.tsv.
Steps:
Load Test Data:
Reads test.tsv using utils.load_data() with the required columns (id, context).
Prediction:
For each row in the test data, extracts features using feature_extraction.extract_features and predicts the name using the trained model.
Save Predictions:
Saves the predicted names in a tab-separated file (submission.tsv), preserving the id for reference.

Main Function:

Purpose: Orchestrates the entire pipeline.
Process:
Ensures the required files and directories exist.
Trains the model on source_unredactor.tsv.
Generates predictions for test.tsv and saves them in submission.tsv.

---

# Prediction Logic (predict.py)
Purpose:
Uses the trained Random Forest model to predict redacted names based on the contextual features extracted from sentences.

Key Function:

predict(model, vec, context, verbose=False)
Input:
model: The trained Random Forest Classifier.

vec: The DictVectorizer used for feature transformation.

context: A single sentence with a redacted portion.

Process:
Extracts features using feature_extraction.extract_features.
Transforms the extracted features into the same format as the training data using vec.

Predicts the name using the model.
Output: The predicted name for the redacted portion of the sentence.

---

# Utility Functions (utils.py)
Purpose:
Handles data loading, validation, and preprocessing to ensure the pipeline operates smoothly.

Key Function:

load_data(file_path, expected_columns=None, is_test=False)
Input:
file_path: Path to the data file.
expected_columns: List of columns expected in the file (e.g., split, name, context for training).
is_test: Boolean indicating whether the file is test data (id and context columns).
Process:
Reads the file line by line to handle rows with extra tabs or malformed data.

Validates that the file contains the required columns.
Output: A cleaned Pandas DataFrame ready for processing.

---

# Pipeline Workflow
Setup:
Directory Setup:
Ensure the required directory structure exists. The data directory should contain the input files (source_unredactor.tsv and test.tsv), and the data directory will store the generated files (unredactor.tsv, unredactor_cleaned.tsv, submission.tsv).

If the directories do not exist, the script will automatically create them.
Input File Validation:
Check that the source_unredactor.tsv file contains three columns: split, name, and context.
Check that the test.tsv file contains two columns: id and context.

Sample Data Creation:
If source_unredactor.tsv is not found, a sample training file with synthetic data is generated to allow users to test the pipeline without actual data.

Training:
Load Training Data:

The source_unredactor.tsv file is read using the load_data function. Rows are validated, and only those with the correct structure (split, name, context) are retained.
The dataset is split into two subsets:
Training Set: Contains rows where split is "training". This subset is used to train the machine learning model.
Validation Set: Contains rows where split is "validation". This subset is used to evaluate the model's performance after training.

Feature Extraction:

Each sentence in the training set is processed to extract meaningful contextual features using the extract_features function.

Vectorization:

The extracted features are transformed into numerical vectors using scikit-learn's DictVectorizer. This step converts the dictionary of features into a matrix that the machine learning model can understand.

Train the Model:

A RandomForestClassifier is trained using the vectorized features and their corresponding labels.
The classifier learns patterns in the data, such as how the context words relate to the redacted name.
Random Forest is chosen for its robustness, ability to handle categorical data, and low likelihood of overfitting with proper parameter tuning.

Validation:
Evaluate the Model:

The model is tested on the validation set to measure its performance on unseen data.
For each sentence in the validation set:
Extract features using extract_features.
Vectorize the features using the same DictVectorizer used during training.
Predict the redacted name using the trained RandomForestClassifier.
Metrics such as accuracy, precision, recall, and F1-score are calculated to assess the model's performance.

Testing:
Load Test Data:

The test.tsv file is read using load_data. Each row contains:
id: A unique identifier for the test sentence.
context: A sentence with a redacted portion.
Rows are validated to ensure they meet the required format.

Predict Missing Names:

For each sentence in the test set:
Extract features using extract_features to capture the context around the redacted portion.
Transform the features into a numerical vector using the trained DictVectorizer.
Use the trained RandomForestClassifier to predict the missing name.

---


# Why This Pipeline is Effective
Scalability: Handles large datasets efficiently by vectorizing features and leveraging Random Forest's parallel processing capabilities.

Flexibility: Can be extended to include additional features or use more advanced models if needed.

Intermediate Results: The inclusion of unredactor.tsv and unredactor_cleaned.tsv allows users to debug and refine the model.

---

# Assumptions

Data Formatting:
source_unredactor.tsv must have three columns: split, name, context.
test.tsv must have two columns: id, context.

Feature Dependence: The model relies on neighboring words and redaction length for predictions.

Model Generalization: The training data sufficiently represents the test data.

---

# Limitations

Context Dependency: Poor context in sentences can lead to incorrect predictions.

Data Bias: Imbalanced or limited training data can affect accuracy.

Computational Resources: Large datasets may require additional computational resources.





