''' This program models the random forest, experiments with hyperparameter tuning, and evaluates the performance'''

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_text
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import matplotlib.pyplot as plt
from get_data import get_data_target

# run model
def model(X_train, X_test, y_train, y_test, list_features):
    # create rf regressor with 100 estimators as default and fit
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)

    # obtain predictions
    y_pred = rf.predict(X_test)

    # display best tree and evaluate
    evaluate(y_test, y_pred)
    plot(rf.estimators_[0], y_test, y_pred, list_features)

# function to conduct hypertuning and run model based on best params
def hyper_tune(X_train, X_test, y_train, y_test, list_features):
    # create grid of hyperparameters
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    # create rf regressor
    rf = RandomForestRegressor()

    # perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print(best_params)

    # Get the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    evaluate(y_test, y_pred)
    plot(best_model.estimators_[0], y_test, y_pred, list_features)
    
# evaluate function
def evaluate(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")
    r2 = r2_score(y_test, y_pred)
    print(f"R2 Score: {r2}\n")

# plot
def plot(mdl, y_test, y_pred, list_features):
    tree_rules = export_text(mdl, feature_names=list_features)
    print(tree_rules)

    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Population Growth (%)")
    plt.ylabel("Predicted Population Growth (%)")
    plt.title("Actual vs. Predicted Population Growth") 
    plt.show()

def main():
    X, y, list_features = get_data_target()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model(X_train, X_test, y_train, y_test, list_features)

    hyper_tune(X_train, X_test, y_train, y_test, list_features)



if __name__ == "__main__":
    main()