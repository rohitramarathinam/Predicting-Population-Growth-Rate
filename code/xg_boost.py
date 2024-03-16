''' This program models XGBoost, experiments with hyperparameter tuning, and evaluates the performance'''

import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from get_data import get_data_target

def main():
    X, y, _ = get_data_target()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert the dataset to DMatrix format, which is the native format for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set hyperparameters
    params = {
        'learning_rate': 0.1,
        'max_depth': 3,
        'alpha': 0.1,
    }

    # Train the XGBoost model
    xgb_model = xgb.train({}, dtrain, num_boost_round=100)

    # Make predictions on the test set
    y_pred = xgb_model.predict(dtest)

    # evaluate
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    r2 = r2_score(y_test, y_pred)
    print(f"R2 Score: {r2}")

    # plot results
    plot(y_test, y_pred)

# plot
def plot(y_test, y_pred):
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Population Growth (%)")
    plt.ylabel("Predicted Population Growth (%)")
    plt.title("Actual vs. Predicted Population Growth") 
    plt.show()


if __name__ == "__main__":
    main()