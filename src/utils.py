import pandas as pd
import os

def load_data(file_path, expected_columns=None, is_test=False):
    """
    Loading and validate the data from the TSV file.
    Handles both training data (3 columns) and test data (2 columns).
    
    Args:
        file_path: Path to the TSV file
        expected_columns: List of expected column names
        is_test: Boolean indicating if this is test data
    """
    print(f"Loading and validating data from {file_path}...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    try:
        # Reading the file as text first
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Processing lines manually to handle tabs in context
        processed_data = []
        for line in lines:
            parts = line.strip().split('\t')
            if is_test:
                # For test data: expecting id and context
                if len(parts) > 2:
                    id_val = parts[0]
                    context = '\t'.join(parts[1:])
                    processed_data.append([id_val, context])
                else:
                    processed_data.append(parts)
            else:
                # For training data: expecting split, name, and context
                if len(parts) > 3:
                    split_type = parts[0]
                    name = parts[1]
                    context = '\t'.join(parts[2:])
                    processed_data.append([split_type, name, context])
                else:
                    processed_data.append(parts)
        
        # Creating DataFrame with appropriate columns
        if is_test:
            data = pd.DataFrame(processed_data, columns=['id', 'context'])
        else:
            data = pd.DataFrame(processed_data, columns=['split', 'name', 'context'])
        
        # Validating columns if expected_columns provided
        if expected_columns:
            missing_columns = set(expected_columns) - set(data.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
        
        if data.empty:
            raise ValueError("The data file is empty")
        
        # Cleaning the data
        data = data.dropna()
        
        print(f"Successfully loaded {len(data)} rows.")
        return data
    except Exception as e:
        raise Exception(f"Error loading the file '{file_path}': {str(e)}")
