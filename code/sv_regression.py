''' This program models support vector regression, experiments with feature scaling, selection, and hyperparameter tuning, and evaluates the performance'''

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from get_data import get_data_target

def model(X_train, X_test, y_train, y_test):
    svr = SVR()
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)

    # evaluate using mse, r2, mae and plot
    evaluate(y_test, y_pred)
    plot(y_test, y_pred)

# feature engineering: scale data
def scale(X_train, X_test):
    # use standard scaler to fit transformed training data and transform the test set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # return scaled data
    return X_train_scaled, X_test_scaled

# feature engineering: select best features
def select(X_train, X_test, y_train):
    # selector = SelectKBest(score_func=f_regression, k=5) 
    # X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    # X_test_selected = selector.transform(X_test_scaled)

    # use SelectKBest to select 5 best features to predict and fit to training data
    selector = SelectKBest(score_func=mutual_info_regression, k=7)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # obtain the features indices and print
    selected_features_indices = selector.get_support(indices=True)
    print(selected_features_indices)

    # return selected data
    return X_train_selected, X_test_selected

# hypertuning parameters: tune hyperparams of svreg model to see if performance improves
def hyper_tune(X_train, X_test, y_train, y_test):
    # create grid of parameters that need to be tuned
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1, 1, 10],
        'epsilon': [0.1, 0.2, 0.5]
    }

    # create regressor
    _svr = SVR(kernel='rbf')

    # perform grid search and obtain best parameters
    grid_search_sv_reg = GridSearchCV(_svr, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search_sv_reg.fit(X_train, y_train)
    best_params_sv_reg = grid_search_sv_reg.best_params_

    print(best_params_sv_reg)

    # run the model with the best hyperparameters
    svr = SVR(**best_params_sv_reg)
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)

    # evaluate using mse, r2, mae and plot
    evaluate(y_test, y_pred)
    plot(y_test, y_pred)

# evaluate function
def evaluate(y_test, y_pred):
    # evaluate using mse, mae, r2
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")
    r2 = r2_score(y_test, y_pred)
    print(f"R2 Score: {r2}\n")

# plot
def plot(y_test, y_pred):
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Population Growth (%)")
    plt.ylabel("Predicted Population Growth (%)")
    plt.title("Actual vs. Predicted Population Growth") 
    plt.show()

def main():
    # obtain data and target
    X, y, _ = get_data_target()

    # do train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # run model as is
    model(X_train, X_test, y_train, y_test)

    # scale data, then run model
    X_train, X_test = scale(X_train, X_test)
    model(X_train, X_test, y_train, y_test)

    # select best features, then run model
    X_train, X_test = select(X_train, X_test, y_train)
    model(X_train, X_test, y_train, y_test)

    # try hyperparamater tuning and see if results improve
    hyper_tune(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()