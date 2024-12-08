import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

def split_data(input_path, output_path, test_size=0.2, random_state=42):
    # Load dataset
    data = pd.read_csv(input_path)

    # Define features and target
    X = data.drop(columns=['Motivation_Level', 'Parental_Involvement', 'Hours_Studied'])
    y = data['Exam_Score']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Save splits to a pickle file
    joblib.dump((X_train, X_test, y_train, y_test), output_path)
    print(f"Data splits saved to {output_path}")

if __name__ == "__main__":
    input_path = "data/processed/processed_data.csv"
    output_path = "data/splits/data_splits.pkl"

    split_data(input_path, output_path)
