from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_and_evaluate_linear_model(splits_path, model_path):
    # Load the data splits
    X_train, X_test, y_train, y_test = joblib.load(splits_path)

    # Initialize and train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Linear Regression - Mean Squared Error: {mse}")
    print(f"Linear Regression - R-squared: {r2}")

    # Save the model
    joblib.dump(model, model_path)
    print(f"Linear Regression model saved to {model_path}")

if __name__ == "__main__":
    splits_path = "data/splits/data_splits.pkl"
    model_path = "models/linear_model.pkl"

    train_and_evaluate_linear_model(splits_path, model_path)
