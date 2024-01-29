#Company Bankruptcy Prediction 
###Estelle Yang


##Introduction
In this project, the aim is to predict company bankruptcy using a provided dataset. To achieve this, the dataset will be split into training and validation sets, and techniques will be applied to deal with imbalanced data. Three classification methods will be selected and their hyperparameters will be tuned. The performance of these models will be evaluated on the validation set, and the results will be used to identify the best model for predicting bankruptcy. All of these processes will be accomplished using Python.


##Data Processing
The train data consists of 5455 samples and 97 variables. The target variable in the training data is binary, with a value of either 0 or 1. While there are no missing values in the training data, there are a few extreme values that need to be dealt with. To address this issue, I replaced extreme values in each variable with the average value. 
Considering the big number of variables, to reduce the risk of overfitting and also make the calculation less complex, feature selection is necessary. In this case, I choose Boruta to perform feature selection on the train data. One reason is that Boruta can handle complex datasets with interdependent features and can identify important features even if they are correlated with other features. And another reason is it can be used with any classification model. (Kursa & Rudnicki, Feature selection with the Boruta package 2010) After running 100 iteration, Boruta choose 53 features out of 96.
In order to check the accuracy and tune the hyperparameters of the model, I split the train data into training (80%) and validation (20%). Additionally, the traning dataset is imbalanced, with a much larger number of non-bankrupt companies compared to bankrupt companies. To address this issue, I used the SMOTE tool to do the oversampling in the training model. SMOTE can create new synthetic samples that are similar to existing minority class samples instead of simply duplicating.


##Model selection and hyperparameter tuning
The dataset is high dimensional, and the target variable is binary and imbalanced, so the model needs to have the ability to capture non-linear decision boundaries and handle high dimension classification tasks. Therefore, I have chosen Random Forest, KNN, and Logistic Regression to perform the classification.  
At the beginning, I compared each training model with rare data and the train data after oversampling. The results showed that using oversampling data to train the model can achieve better results. Therefore, I used oversampling for further training.
To obtain higher accuracy, AUC, and F1 score, I need to optimize the model for the training data by tuning the hyperparameters. For KNN, the choice of the hyperparameter k can be the most important, so I will try the K value from 1 to 20 and then choose the K value with the highest F1 score. For Random Forest and Logistic Regression, I will first set a range of their important hyperparameters. In Random Forest, the most important two features are the number of decision trees in the forest and the number of features considered by each tree when splitting a node, so I will set 'n_estimators’: [200, 400, 600, 800] and 'max_features’: ['sqrt', 'log2', "auto", 10]. The main hyperparameters I may tune in Logistic Regression are C and penalty, so I will set two penalties, which are “l1” and “l2,” and 5 C values. Then, I will import GridSearchCV from Scikit-Learn to perform all the model settings with 10-fold CV automatically and return the best combination of the hyperparameters.

##Results
For Random Forest, with the best hyperparameters returned by GridSearchCV: {'max_features': 10, 'n_estimators': 200}, the accuracy score for validation is 97.07% and the auc is 81.82%. The F1 score for 0 class is 0.98 and for 1 class is 0.57, with an overall F1 score of 0.5676.
For KNN, the k value with the highest F1 score is 2. The KNN model with n_neighbors=2 return the accuracy score 94.22% and auc 74.30%. The F1 score for 0 class is 0.97 and for 1 class is 0.35 while the entire F1 score is 0.3505.
For logistic regression, I have the best hyperparameters: {'C': 100, 'penalty': 'l2'}. The accuracy score for validation is 89.92% and the auc is 88.75%. The F1 score for 0 class is 0.95 and for 1 class is 0.34, with an overall F1 score of 0.3373. 


##Model Selection
From the results, we can see that Random Forest performs the best in terms of accuracy and has a much higher F1 score than the other two classification models. Meanwhile, the logistic regression model provides the best AUC score and ROC curve. AUC measures the model's ability to correctly classify positive and negative cases, regardless of the threshold used to make the classification decision. In contrast, the F1 score takes into account both precision and recall, which are important measures for imbalanced datasets. Given that the model is used for prediction, to correctly identify positive cases (true positives) and minimize false negatives, I have chosen the Random Forest model for further predictions based on its F1 score.


##Prediction
The test data has 1364 samples and 96 variables. I applied the same feature selection process to the test data and put the filtered data into the random forest model with the best parameters obtained from GridsearchCV. The outcome for bankruptcy shows 1298 zero values and 66 one values.

##Reference
Kursa, M. B., & Rudnicki, W. R. (n.d.). Feature selection with the Boruta package. Journal of Statistical Software. Retrieved September 2010, from https://www.jstatsoft.org/article/view/v036i11 
![image](https://github.com/Estellewwww/Company-Bankruptcy-Prediction/assets/118770424/1edd69e4-d861-4c47-b032-da9df45479b9)
