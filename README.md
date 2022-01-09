# Credit Risk Analysis Using Python Machine Learning
Credit Risk using supervised machine learning approaches for an imbalanced dataset

## Overview
Using a dataset from a peer to peer lending group called LendingClub, it was desired to predict credit risk. The dataset is an imbalanced set due to the large number of low risk records as compared to high risk.  Several approaches can be taken on this type of imbalanced data.  Such approaches as undersampling the low risk records or oversampling the high risk records for the training set are evaluated.  In addition, algorithms designed specifically for imbalanced datasets are also leveraged.

## Results

The data was prepped by converting strings to numbers using the get_dummies method from the Pandas library.  A total of 95 features comprised the independent variables with a target dependent variable of high risk or low risk loans.  Such independent variables included things like home ownership, income, loan payment amounts, interest, income, etc.   When splitting the data into training and testing data with the imbalanced data only 260 high risk loans were in the training sample with 51,352 low risk loans.  For the testing set this left 87 high risk loans and 17,118 low risk loan records.

As seen in the table below, four oversampling and under sampling techniques were leveraged with Logistic Regression machine learning.   Both the Naive and SMOTE (Synthetic Minority Oversampling Technique) had similar results.  While the SMOTEENN sampling method using a combination of SMOTE and edited nearest neighbor was a further improvement.  The machine learning algorithm of Balanced Random Forest has internal sampling strategies for each tree built-in, but showed degredation in the recall.  Lastly, the Adaboost had the best performance with both accruacy and recall over 90%.

![alt text](https://github.com/jj2773/Credit_Risk_Analysis/blob/main/summary_table.PNG)



## Summary
As explained in the confusion matrix graphic below, the recall is 91% which means that this algorithm predicted 91% of the actual high risk loans. All algorithms showed poor performance in precision which means that the number of false positive is very high.  Although the Adaboost performed the best an approach to address the false positives is needed.


![alt text](https://github.com/jj2773/Credit_Risk_Analysis/blob/main/confusion_matrix.PNG)