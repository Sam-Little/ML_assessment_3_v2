from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_and_evaluate_rf_model(splits_path, model_path, random_state=33):
    # Load the data splits
    X_train, X_test, y_train, y_test = joblib.load(splits_path)

    # Initialize and train the Random Forest model
    rf_model = RandomForestRegressor(random_state=random_state)
    rf_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Random Forest - Mean Squared Error: {mse}")
    print(f"Random Forest - R-squared: {r2}")

    # Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure the directory exists
    joblib.dump(rf_model, model_path)
    print(f"Random Forest model saved to {model_path}")

if __name__ == "__main__":
    # Paths relative to the script's location
    current_dir = os.path.dirname(__file__)
    splits_path = os.path.join(current_dir, "../../data/splits/data_splits.pkl")
    model_path = os.path.join(current_dir, "../../models/random_forest_model.pkl")

    train_and_evaluate_rf_model(splits_path, model_path)
