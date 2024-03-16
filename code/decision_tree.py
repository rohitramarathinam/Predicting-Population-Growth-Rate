''' This program models the decision tree, experiments with hyperparameter tuning, and evaluates the performance'''

from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from get_data import get_data_target

# run model
def model(X_train, X_test, y_train, y_test, list_features):
    # create regressor, fit, obtain predictions
    dt = DecisionTreeRegressor()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    # evaluate using mse, r2, mae and plot the tree
    evaluate(y_test, y_pred)
    plot(dt, y_test, y_pred, list_features)

# function to conduct hypertuning and run model based on best params
def hyper_tune(X_train, X_test, y_train, y_test, list_features):
    # create grid of hyperparameters 
    param_grid = {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # create regressor
    dt = DecisionTreeRegressor()

    # perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print(best_params)

    # Get the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # evaluate using mse, r2, mae and plot
    evaluate(y_test, y_pred)
    plot(best_model, y_test, y_pred, list_features)

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

    # run default model
    model(X_train, X_test, y_train, y_test, list_features)

    # run hypertuned model
    hyper_tune(X_train, X_test, y_train, y_test, list_features)


if __name__ == "__main__":
    main()