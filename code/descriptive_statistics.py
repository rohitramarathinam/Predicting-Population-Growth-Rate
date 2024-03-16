''' This program returns descriptive statistics of each feature in the dataset and reports the top 10 and worst 10 countries for each feature'''

import json
import numpy as np

def main():
    # reading json file and loading dataset
    with open('dataset.json', 'r') as file:
        dataset = json.load(file)

    # reading features in
    with open('list_features.txt', 'r') as file:
        list_features = [line.strip() for line in file]

    list_features.append("People and Society: Population growth rate")

    # create lists holding all data sorted
    lists = []

    # append by feature name and sort highest to lowest
    for feature in list_features:
        t_list = []
        for country, factors in dataset.items():
            t_list.append({country: factors.get(feature)})
        sorted_list = sorted(t_list, key=lambda x: list(x.values())[0], reverse=True)
        lists.append(sorted_list)

    # report top 10
    i = 0
    for l in lists:
        print(f"Countries with highest \"{list_features[i]}\":")
        for j in range(10):
            print(f"{j+1}. {l[j]}")

        print(f"\nCountries with lowest \"{list_features[i]}\":")

        for k in range(len(l)-1, len(l)-11, -1):
            print(f"{k+1}. {l[k]}")
        
        # get numerical values
        values = [list(country.values())[0] for country in l]

        # calculate descriptive statistics
        mean_value = np.mean(values)
        median_value = np.median(values)
        std_dev = np.std(values)

        # print
        print(f"\nMean: {mean_value}")
        print(f"Median: {median_value}")
        print(f"Standard Deviation: {std_dev}")

        print("\n------------------------------------------------\n")
        i+=1





if __name__ == "__main__":
    main()