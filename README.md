# Company Bankruptcy Prediction

**Imbalanced classification · Boruta feature selection · SMOTE · Model comparison**

An MF815 academic project comparing random forest, K-nearest neighbors, and logistic regression for company-bankruptcy classification using the provided financial dataset.

## Explore

| Resource | Purpose |
| --- | --- |
| [Analysis notebook](Company_Bankruptcy_Prediction.ipynb) | Preprocessing, feature selection, resampling, tuning, and evaluation |
| [Project report](Company%20Bankruptcy%20Prediction.pdf) | Original written analysis |
| [Training data](train.csv) / [Test data](test.csv) | Provided project inputs |

## Workflow

Data cleaning → Boruta feature selection → training/validation split → SMOTE → classifier comparison → test-set predictions.

The original write-up reports validation accuracy, F1, and an AUC calculation. The notebook computes AUC from hard class predictions rather than probability scores. Feature selection is also performed before the validation split, and oversampling occurs before cross-validation, so those historical metrics should not be treated as independently validated generalization estimates.

## Setup & scope

Python with pandas, NumPy, Matplotlib, scikit-learn, imbalanced-learn, and Boruta. The notebook includes machine-specific file paths, legacy APIs, and cell-order dependencies that need adjustment before a clean run. The original code and outputs are preserved; they have not been rerun as part of this documentation update.

For a future modeling revision, fit preprocessing and feature selection only on training folds, keep SMOTE inside each cross-validation fold, and evaluate probability-based ROC/PR metrics on an untouched holdout.

<details>
<summary><strong>Original project write-up and reported results</strong></summary>

# Company Bankruptcy Prediction 
### Estelle Yang


## Introduction
In this project, the aim is to predict company bankruptcy using a provided dataset. To achieve this, the dataset will be split into training and validation sets, and techniques will be applied to deal with imbalanced data. Three classification methods will be selected and their hyperparameters will be tuned. The performance of these models will be evaluated on the validation set, and the results will be used to identify the best model for predicting bankruptcy. All of these processes will be accomplished using Python.


## Data Processing
The train data consists of 5455 samples and 97 variables. The target variable in the training data is binary, with a value of either 0 or 1. While there are no missing values in the training data, there are a few extreme values that need to be dealt with. To address this issue, I replaced extreme values in each variable with the average value. 

Considering the large number of variables, to reduce the risk of overfitting and also make the calculation less complex, feature selection is necessary. In this case, I choose Boruta to perform feature selection on the train data. One reason is that Boruta can handle complex datasets with interdependent features and can identify important features even if they are correlated with other features. And another reason is it can be used with any classification model. (Kursa & Rudnicki, Feature selection with the Boruta package 2010) After running 100 iterations, Boruta chooses 53 features out of 96.

To check the accuracy and tune the hyperparameters of the model, I split the training data into training (80%) and validation (20%). Additionally, the training dataset is imbalanced, with a much larger number of non-bankrupt companies compared to bankrupt companies. To address this issue, I used the SMOTE tool to do the oversampling in the training model. SMOTE can create new synthetic samples that are similar to existing minority class samples instead of simply duplicating them.


## Model selection and hyperparameter tuning
The dataset is highly dimensional, and the target variable is binary and imbalanced, so the model must be able to capture non-linear decision boundaries and handle high-dimension classification tasks. Therefore, I have chosen Random Forest, KNN, and Logistic Regression to perform the classification.  

Initially, I compared each training model with rare data and the training data after oversampling. The results showed that using oversampling data to train the model can achieve better results. Therefore, I used oversampling for further training.
To obtain higher **accuracy**, **AUC**, and **F1** scores, I need to optimize the model for the training data by tuning the hyperparameters. For KNN, the choice of the hyperparameter k can be the most important, so I will try the K value from 1 to 20 and then choose the K value with the highest F1 score. For Random Forest and Logistic Regression, I will first set a range of their important hyperparameters. In Random Forest, the most important two features are the number of decision trees in the forest and the number of features considered by each tree when splitting a node so I will set *'n_estimators’: [200, 400, 600, 800]* and *'max_features’: ['sqrt', 'log2', "auto", 10]*. The main hyperparameters I may tune in Logistic Regression are C and penalty, so I will set two penalties, which are “l1” and “l2,” and 5 C values. Then, I will import GridSearchCV from Scikit-Learn to perform all the model settings with 10-fold CV automatically and return the best combination of the hyperparameters.

## Results
For Random Forest, with the best hyperparameters returned by GridSearchCV: {'max_features': 10, 'n_estimators': 200}, the accuracy score for validation is **97.07%** and the AUC is **81.82%**. The F1 score for 0 class is **0.98** and for 1 class is **0.57**, with an overall F1 score of **0.5676**.

For KNN, the k value with the highest F1 score is 2. The KNN model with n_neighbors=2 returns an accuracy score of **94.22%** and an AUC of **74.30%**. The F1 score for the 0 class is **0.97** and for 1 class is **0.35** while the entire F1 score is **0.3505**.

For logistic regression, I have the best hyperparameters: {'C': 100, 'penalty': 'l2'}. The accuracy score for validation is **89.92%** and the AUC is **88.75%**. The F1 score for the 0 class is **0.95** and for the 1 class is **0.34**, with an overall F1 score of **0.3373**. 


## Model Selection
From the results, we can see that Random Forest performs the best in terms of accuracy and has a much higher F1 score than the other two classification models. Meanwhile, the logistic regression model provides the best AUC score and ROC curve. AUC measures the model's ability to correctly classify positive and negative cases, regardless of the threshold used to make the classification decision. In contrast, the F1 score takes into account both precision and recall, which are important measures for imbalanced datasets. Given that the model is used for prediction, to correctly identify positive cases (true positives) and minimize false negatives, I have chosen the Random Forest model for further predictions based on its F1 score.


## Prediction
The test data has 1364 samples and 96 variables. I applied the same feature selection process to the test data and put the filtered data into the random forest model with the best parameters obtained from GridsearchCV. The outcome for bankruptcy shows 1298 zero values and 66 one values.

## Reference
Kursa, M. B., & Rudnicki, W. R. (n.d.). Feature selection with the Boruta package. Journal of Statistical Software. Retrieved September 2010, from https://www.jstatsoft.org/article/view/v036i11 
![image](https://github.com/Estellewwww/Company-Bankruptcy-Prediction/assets/118770424/1edd69e4-d861-4c47-b032-da9df45479b9)


</details>
