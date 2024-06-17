from sklearn import datasets 
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

############## FOR EVERYONE ##############
# Please note that the blanks are here to guide you for this first assignment, but the blanks are  
# in no way representative of the number of commands/ parameters or length of what should be inputted.

### PART 1 ###
# Scikit-Learn provides many popular datasets. The breast cancer wisconsin dataset is one of them. 
# Write code that fetches the breast cancer wisconsin dataset. 
# Hint: https://scikit-learn.org/stable/datasets/toy_dataset.html
# Hint: Make sure the data features and associated target class are returned instead of a "Bunch object".
X, y = datasets.load_breast_cancer(return_X_y=True) #(5 points) 

# Check how many instances we have in the dataset, and how many features describe these instances
print("There are",len(X), "instances described by", len(X[0]), "features.") #(5 points)  

# Create a training and test set such that the test set has 40% of the instances from the 
# complete breast cancer wisconsin dataset and that the training set has the remaining 60% of  
# the instances from the complete breast cancer wisconsin dataset, using the holdout method. 
# In addition, ensure that the training and test sets # contain approximately the same 
# percentage of instances of each target class as the complete set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state = 42)  #(5 points) 

# Create a decision tree classifier. Then Train the classifier using the training dataset created earlier.
# To measure the quality of a split, using the entropy criteria.
# Ensure that nodes with less than 6 training instances are not further split
clf = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=6)   #(5 points) 
clf = clf.fit(X_train, y_train)  #(4 points) 

# Apply the decision tree to classify the data 'testData'.
predC = clf.predict(X_test)  #(4 points) 

# Compute the accuracy of the classifier on 'testData'
print('The accuracy of the classifier is', accuracy_score(y_test, predC))  #(3 points) 

# Visualize the tree created. Set the font size the 12 (5 points) 
# tree = plt.plot(X_train,y_train)  
_ = tree.plot_tree(clf, 
                   feature_names=datasets.load_breast_cancer().feature_names,  
                   class_names=datasets.load_breast_cancer().target_names,
                   filled=True,
                   fontsize=12)

### PART 2.1 ###
# Visualize the training and test error as a function of the maximum depth of the decision tree
# Initialize 2 empty lists where you will save the training and testing accuracies 
# as we iterate through the different decision tree depth options.
trainAccuracy = [] #(1 point) 
testAccuracy = [] #(1 point) 
# Use the range function to create different depths options, ranging from 1 to 15, for the decision trees
depthOptions = range(1, 16) #(1 point) 
for depth in depthOptions: #(1 point) 
    # Use a decision tree classifier that still measures the quality of a split using the entropy criteria.
    # Also, ensure that nodes with less than 6 training instances are not further split
    cltree = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=6, max_depth=depth) #(1 point) 
    # Decision tree training
    cltree = cltree.fit(X_train, y_train) #(1 point) 
    # Training error
    y_predTrain = cltree.predict(X_train) #(1 point) 
    # Testing error
    y_predTest = cltree.predict(X_test) #(1 points) 
    # Training accuracy
    trainAccuracy.append(accuracy_score(y_train, y_predTrain)) #(2 points) 
    # Testing accuracy
    testAccuracy.append(accuracy_score(y_test, y_predTest)) #(2 points) 

# Plot of training and test accuracies vs the tree depths (use different markers of different colors)
plt.plot(list(depthOptions), trainAccuracy, 'rx', list(depthOptions), testAccuracy, 'b-.') #(3 points) 
plt.legend(['Training Accuracy','Test Accuracy']) # add a legend for the training accuracy and test accuracy (1 point) 
plt.xlabel('Tree Depth') # name the horizontal axis 'Tree Depth' (1 point) 
plt.ylabel('Classifier Accuracy') # name the vertical axis 'Classifier Accuracy' (1 point) 

# Fill out the following blanks: #(6 points (3 points per blank)) 
""" 
According to the test error, the best model to select is when the maximum depth is equal to 3, approximately. 
But, we should not use select the hyperparameters of our model using the test data, because this is considered _____.
"""

### PART 2.2 ###
# Use sklearn's GridSearchCV function to perform an exhaustive search to find the best tree depth.
# Hint: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# First define the parameter to be optimized: the max depth of the tree
parameters = {"max_depth":list(depthOptions)} #(6 points)
# We will still grow a decision tree classifier by measuring the quality of a split using the entropy criteria. 
# Also, continue to ensure that nodes with less than 6 training instances are not further split.
clf = GridSearchCV(tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=6), parameters, refit=True) #(6 points)
clf = clf.fit(X_train, y_train) #(5 points)
tree_model = clf.best_estimator_ #(5 points)
print("The maximum depth of the tree should be", clf.best_params_) #(5 points)

# The best model is tree_model. Visualize that decision tree (tree_model). Set the font size the 12 
_ = tree.plot_tree(tree_model, 
                   feature_names=datasets.load_breast_cancer().feature_names,  
                   class_names=datasets.load_breast_cancer().target_names,
                   filled=True,
                   fontsize=12) #(5 points)

# Fill out the following blank: #(3 points)
""" 
This method for tuning the hyperparameters of our model is acceptable, because ________. 
"""

# Explain below was is tenfold Stratified cross-validation?  #(5 points)
"""
______
"""

############## FOR GRADUATE STUDENTS ONLY (the students enrolled in CPS 8318) ##############
### PART 3: decision tree classifier to solve a classification problem ###
""" 
Please note that the grade for parts 1 and 2 counts for 75% of your total grade. The following
work requires you to work on a project of your own and will account for the remaining 25% of your grade.

Choose a practical dataset (as opposed to the example ones we used in class) with a reasonable size 
from one of the following sources (other sources are also possible, e.g., Kaggle):

•	UCI Machine Learning Repository, https://archive.ics.uci.edu/ml/datasets.php. 

•	KDD Cup challenges, http://www.kdd.org/kdd-cup.

Download the data, read the description, and use a decision tree classifier approach to solve a 
classification problem as best as you can. Write up a report of approximately 1 page, double spaced, 
in which you briefly describe the dataset (e.g., the size – number of instances and number of attributes, 
what type of data, source), the problem, the approaches that you tried and the results. 
You can use any appropriate libraries. 

Your tasks are:
1.	research how to pre-process data on your own, if needed by the dataset you chose.
2.	to report on which attributes are most important for your classifier 
(hint: the feature that gives you the most information gain about the class labels).
3.	to report on anything else inventive you can think to do, but the above 2 tasks would be enough. 

Marking: Part 3 accounts for 25% of your final grade. 50% of that will be for the writeup and 50% 
for the results. In the write-up, cite the sources of your data and ideas, and use your own words 
to express your thoughts. If you have to use someone else's words or close to them, use quotes and 
a citation.  The citation is a number in brackets (like [1]) that refers to a similar number in the 
references section at the end of your paper or in a footnote, where the source is given as an author, 
title, URL or journal/conference/book reference. Grammar is important. Concerning the 50% for results, 
elaborate on what (if any) manipulations you did, what are the results for the algorithms you tried, 
what else you tried. 

Submit the python script (.py file(s)) with your redacted document (PDF file) on the D2L site. 
If the dataset is not in the public domain, you also need to submit the data file. 
Name your documents appropriately:
report_Firstname_LastName.pdf
script_ Firstname_LastName.py
"""