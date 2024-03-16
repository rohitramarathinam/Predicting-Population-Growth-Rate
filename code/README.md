# Predicting Population Growth Rate Trends from Human Geography Factors

## Executing program(s)

* Before you run any of the models, please ensure that you are in the directory "code" and not in "ML Project". Thanks!
* In case you'd like to just see the plots of each model, navigate back to "ML Project" and into "img". Then select the folder with the name of the model whose results you wish to view to see all the plots generated
* There are 12 python files that are a part of this project. These will be detailed below


### Non .py files

* countries.json is the original dataset in json format
* countries.csv is the original dataset in csv format
* dataset.json is the file after filtering out countries and features
* kaggle.json was downloaded when the dataset was originally downloaded from kaggle.com
* list_features.txt is the text file containing all features that were selected through setup.py and manual feature engineering
* README.md is this file; there is another README.md in the previous directory that details an overview of the entire project



### setup.py

* This file's purpose is to setup the dataset and filter out unrecognized countries (by UN) and territories that may belong to other sovereign states (ex. Guam will be removed because it is part of the USA)
* If dataset.json is not already populated, run this file once to populate dataset.json with the necessary features and target



### descriptive_statistics.py

* This file's purpose is to display top 10 and worst 10 countries per each selected factor, and calculate descriptive statistics for each feature
* Run this file if you are interested in learning what countries have the highest and lowest population growth rate, median age, etc.
* This file is purely an informative program and doesn't have any machine learning models implemented



### get_data.py

* This file's purpose is to read from the dataset and obtain {data, target} for the machine learning models to use
* It returns all the features except population growth rate as 'data'
* It returns populaton growth rate as 'target'
* It also returns the features in list format for the tree-based models to use for tree visualization (in terminal)



### linear_regression.py

* This file's purpose is to create a linear regression model, run it on the dataset, and then perform feature scaling, selection, and hyperparameter tuning to see if results improve
* Click run, and the first plot will appear, detailing the actual vs predicted graph
* Close this image for the next to appear
* The images show up in this order: default parameters, scaled features, selected features, hypertuned parameters
* All evaluation results can be found on the terminal and follow the same order as above



### sv_regression.py

* This file's purpose is to create a support vector regression model, run it on the dataset, and then perform feature scaling, selection, and hyperparameter tuning to see if results improve
* Click run, and the first plot will appear, detailing the actual vs predicted graph
* Close this image for the next to appear
* The images show up in this order: default parameters, scaled features, selected features, hypertuned parameters
* All evaluation results can be found on the terminal and follow the same order as above



### ridge_regression.py

* This file's purpose is to create a ridge regression model, run it on the dataset, and then perform feature scaling, selection, and hyperparameter tuning to see if results improve
* Click run, and the first plot will appear, detailing the actual vs predicted graph
* Close this image for the next to appear
* The images show up in this order: default parameters, scaled features, selected features, hypertuned parameters
* All evaluation results can be found on the terminal and follow the same order as above



### lasso_regression.py

* This file's purpose is to create a lasso regression model, run it on the dataset, and then perform feature scaling, selection, and hyperparameter tuning to see if results improve
* Click run, and the first plot will appear, detailing the actual vs predicted graph
* Close this image for the next to appear
* The images show up in this order: default parameters, scaled features, selected features, hypertuned parameters
* All evaluation results can be found on the terminal and follow the same order as above



### decision_tree.py

* This file's purpose is to create a decision tree regression model, run it on the dataset, and then perform hyperparameter tuning to see if results improve
* Click run, and the first plot will appear, detailing the actual vs predicted graph
* Close this image for the next to appear
* The images show up in this order: default parameters, hypertuned parameters
* All evaluation results can be found on the terminal and follow the same order as above
* Addiitonally, the tree rules visualization can be seen on the terminal



### random_forest.py

* This file's purpose is to create a random forest regression model, run it on the dataset, and then perform hyperparameter tuning to see if results improve
* Click run, and the first plot will appear, detailing the actual vs predicted graph
* Close this image for the next to appear
* The images show up in this order: default parameters, hypertuned parameters
* All evaluation results can be found on the terminal and follow the same order as above
* Addiitonally, the tree rules visualization of the best tree can be seen on the terminal



### gradient_boosting.py

* This file's purpose is to create a gradient boosting regression model, run it on the dataset, and then perform hyperparameter tuning to see if results improve
* Click run, and the first plot will appear, detailing the actual vs predicted graph
* Close this image for the next to appear
* The images show up in this order: default parameters, hypertuned parameters
* All evaluation results can be found on the terminal and follow the same order as above
* Addiitonally, the tree rules visualization of the best tree can be seen on the terminal



### knn.py

* This file's purpose is to create a K-Nearest Neighbors regression model, run it on the dataset, and then perform feature scaling, selection, and hyperparameter tuning to see if results improve
* Click run, and the first plot will appear, detailing the actual vs predicted graph
* Close this image for the next to appear
* The images show up in this order: default parameters, scaled features, selected features, hypertuned parameters
* All evaluation results can be found on the terminal and follow the same order as above



### xg_boost.py

* This file's purpose is to create a xg boosting regression model and run it on the dataset
* For this, please ensure that you have installed XGBoost on your machine
* Click run, and the plot will appear, detailing the actual vs predicted graph
* All evaluation results can be found on the terminal
* Addiitonally, the tree rules visualization of the best tree can be seen on the terminal