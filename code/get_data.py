import json
import numpy as np

# read json file, load dataset, and obtain data and target
def get_data_target():
    # reading json file and loading dataset
    with open('dataset.json', 'r') as file:
        dataset = json.load(file)

    # reading in features that will be used to predict
    with open('list_features.txt', 'r') as file:
        list_features = [line.strip() for line in file]
        
    data = []
    target = []

    # parse through all countries in dataset and add all features to data, and target variable to target
    for country in dataset:
        data.append([dataset[country][i] for i in list_features])
        target.append(dataset[country]["People and Society: Population growth rate"])

    # store as numpy arrays
    X = np.array(data)
    y = np.array(target)

    # return data, target
    return X, y, list_features