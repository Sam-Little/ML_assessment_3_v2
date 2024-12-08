import pandas as pd

file_path = 'C:/Users/Admin/ML_assessment_3_v2/data/raw/StudentPerformanceFactors.csv'



def load_data(file_path):
    """
    Load the dataset from a CSV file.
    Parameters:
        file_path (str): Path to the dataset file.
    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    return pd.read_csv(file_path)

def preprocess_data(data):
    """
    Preprocess the dataset by handling missing values.
    Parameters:
        data (pd.DataFrame): Raw dataset.
    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    # Handle missing values by filling with column means
    data = data.fillna(data.mean())
    
    # Additional preprocessing can be added here
    # Example: Encoding categorical variables
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        data[col] = data[col].astype('category').cat.codes  # Convert to numeric codes

    return data

if __name__ == "__main__":
    # Define file paths
    input_file_path = "../../data/raw/StudentPerformanceFactors.csv"
    output_file_path = "../../data/processed/processed_data.csv"

    # Load the data
    print("Loading data...")
    raw_data = load_data(input_file_path)

    # Display the first few rows for inspection
    print("Raw data sample:")
    print(raw_data.head())

    # Preprocess the data
    print("Preprocessing data...")
    processed_data = preprocess_data(raw_data)

    # Save the preprocessed data
    print(f"Saving preprocessed data to {output_file_path}...")
    processed_data.to_csv(output_file_path, index=False)

    print("Data preprocessing complete!")