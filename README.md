# Analysis-with-R
(Completed for a Business Intelligence and Analytics class during undergrad)

Given the “bank.csv” dataset the objective of this experiment is to create a classification algorithm that will predict 
whether a client will subscribe to a term deposit or not. In order to create this classification 
the four following classification algorithms will be used: CTree, J48, linear classification, and k-nearest neighbors (k-nn) classification.

Based on the F1- Score for each classification it is clear that KNN has performed the best since its F1-Score is 0.0246. 
Following this is CTree, J48, and Linear which have all performed very poorly in terms of their dataset and also failed 
to recall most of the “yes” rows from the test data since they have a F1-score of 0.
F1 score is a useful method to compare classifiers than recall, accuracy, or precision since the experiment is data heavy. 
The data set is skewed to one class a lot more with a predominant number of “no”  than “yes”. 
Also if a model has a high recall and low precision or vice versa this will make it difficult to evaluate why we use F1 Score as a main comparison method.
