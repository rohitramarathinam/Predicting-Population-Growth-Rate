''' This program models the KNN model, experiments with feature scaling, selection, and hyperparameter tuning, and evaluates the performance'''

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from get_data import get_data_target

def model(X_train, X_test, y_train, y_test):
    knnr = KNeighborsRegressor()
    knnr.fit(X_train, y_train)

    y_pred = knnr.predict(X_test)

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
    # use SelectKBest to select 5 best features to predict and fit to training data
    selector = SelectKBest(score_func=mutual_info_regression, k=10)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # obtain the features indices and print
    selected_features_indices = selector.get_support(indices=True)
    print(selected_features_indices)

    # return selected data
    return X_train_selected, X_test_selected


def hyper_tune(X_train, X_test, y_train, y_test, list_features):
    param_grid = {
        'n_neighbors': [i+1 for i in range(len(list_features))]
    }

    knnr = KNeighborsRegressor()

    grid_search = GridSearchCV(knnr, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print(best_params)

    # Get the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    evaluate(y_test, y_pred)
    plot(y_test, y_pred)

# evaluate function
def evaluate(y_test, y_pred):
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

# main function to run 
def main():
    # obtain data and target
    X, y, list_features = get_data_target()

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
    hyper_tune(X_train, X_test, y_train, y_test, list_features)


if __name__ == "__main__":
    main()