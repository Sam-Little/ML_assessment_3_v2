import joblib
from sklearn.metrics import mean_squared_error, r2_score
import os

def evaluate_model(model_path, splits_path):
    """
    Evaluate a trained model using test data.

    Parameters:
        model_path (str): Path to the saved model file.
        splits_path (str): Path to the saved data splits file.

    Returns:
        dict: Evaluation metrics (MSE and R²).
    """
    # Load the model
    model = joblib.load(model_path)

    # Load the data splits
    X_train, X_test, y_train, y_test = joblib.load(splits_path)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {"Mean Squared Error": mse, "R-squared": r2}

if __name__ == "__main__":
    # Paths relative to the script's location
    current_dir = os.path.dirname(__file__)
    splits_path = os.path.join(current_dir, "../../data/splits/data_splits.pkl")

    # Evaluate Linear Regression model
    linear_model_path = os.path.join(current_dir, "../../models/linear_model.pkl")
    print("Evaluating Linear Regression Model...")
    linear_metrics = evaluate_model(linear_model_path, splits_path)
    print(f"Linear Regression Metrics: {linear_metrics}")

    # Evaluate Random Forest model
    rf_model_path = os.path.join(current_dir, "../../models/random_forest_model.pkl")
    print("Evaluating Random Forest Model...")
    rf_metrics = evaluate_model(rf_model_path, splits_path)
    print(f"Random Forest Metrics: {rf_metrics}")
