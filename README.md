# Spark-ML
this project is the implementation of Machine Learning Algorithm in Spark to predict the housing price for next year


Machine Learning Libraries

we are going to implement folowing libraries and compare the prediction. Since this is just a sample, we will never know wheter our prediction is accurate or not.

Innetial DataSet: 
    ![](https://github.com/parlad/Spark-ML/blob/master/Images/Dataset.png)

    1. Decision Tree Regression
    2. Random Forest Regression
    3. Gradient Boosted Tree Regression

1. Decision Tree Regression 
Introduction : Decision tree builds regression or classification models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. A decision node (e.g., Outlook) has two or more branches (e.g., Sunny, Overcast and Rainy), each representing values for the attribute tested. Leaf node (e.g., Hours Played) represents a decision on the numerical target. The topmost decision node in a tree which corresponds to the best predictor called root node. Decision trees can handle both categorical and numerical data.  	
                    ![](https://saedsayad.com/images/Decision_tree_r1.png)

Code Snepits : 
```
    #Decision Tree Regression
    decisionTree = DecisionTreeRegressor(featuresCol = "Features", labelCol = "Yearly Amount Spent", maxDepth = 15, maxBins = 32)
    decisionTreeModel = decisionTree.fit(trainingData)
    dtresults = decisionTreeModel.transform(testingData)
    dtresults.select("Prediction", "Yearly Amount Spent", "Features")
    dtresults.show()
```

Prediction Outcome : 
![](https://github.com/parlad/Spark-ML/blob/master/Images/decisionTreeRegressionoutput.png)

2. Random Forest Regression: 
Introduction : A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default). A Random Forest is an ensemble technique capable of performing both regression and classification tasks with the use of multiple decision trees and a technique called Bootstrap Aggregation, commonly known as bagging. What is bagging you may ask? Bagging, in the Random Forest method, involves training each decision tree on a different data sample where sampling is done with replacement.
        ![](https://github.com/parlad/Spark-ML/blob/master/Images/randomForest.png)

Code Snepits :
```
#Random Forest Regression
randomForest = RandomForestRegressor(featuresCol = "Features", labelCol = "Yearly Amount Spent",  maxDepth = 15, maxBins = 32, numTrees = 200)
randomForestModel = randomForest.fit(trainingData)
rfresults = randomForestModel.transform(testingData)
rfresults.select("Prediction", "Yearly Amount Spent", "Features")
rfresults.show()
```

Prediction Outcome : 
![](https://github.com/parlad/Spark-ML/blob/master/Images/RandomeForestRegresionOutput.png)

3. Gradient Boosted Tree Regression
Introduction :  Boosting is a method of converting weak learners into strong learners. In boosting, each new tree is a fit on a modified version of the original data set. The gradient boosting algorithm (gbm) can be most easily explained by first introducing the AdaBoost Algorithm.The AdaBoost Algorithm begins by training a decision tree in which each observation is assigned an equal weight. After evaluating the first tree, we increase the weights of those observations that are difficult to classify and lower the weights for those that are easy to classify. The second tree is therefore grown on this weighted data. Here, the idea is to improve upon the predictions of the first tree. Our new model is therefore Tree 1 + Tree 2. We then compute the classification error from this new 2-tree ensemble model and grow a third tree to predict the revised residuals. We repeat this process for a specified number of iterations. Subsequent trees help us to classify observations that are not well classified by the previous trees. Predictions of the final ensemble model is therefore the weighted sum of the predictions made by the previous tree models.

Code Snepits :
```
boostedTree = GBTRegressor(featuresCol = "Features", labelCol = "Yearly Amount Spent", maxDepth = 5, maxBins = 32, maxIter = 200)
boostedTreeModel = boostedTree.fit(trainingData)
gbtresults = boostedTreeModel.transform(testingData)
gbtresults.select("Prediction", "Yearly Amount Spent", "Features")
gbtresults.show()
```
Prediction Outcome : 
![](https://github.com/parlad/Spark-ML/blob/master/Images/Gradient_Boosted_Tree%20Regression_outcome.png)




