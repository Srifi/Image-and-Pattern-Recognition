#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 00:05:10 2021
Last modified on Tue Dec 1 2022

@author: SRIFI NAJLAA

"""

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import preprocessing
import numpy as np
import h5py
import matplotlib.pyplot as plt
from math import exp, log
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from keras.utils import to_categorical
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import pandas as pd
from pandas import plotting
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import  BatchNormalization
from keras.constraints import maxnorm

def calculate_cost_LogReg(y, y_hat):
    """
    Calculates the cost of the OUTPUT OF JUST ONE pattern from the logistic
    regression classifier (i.e. the result of applying the h function) and
    its real class.
    
    Parameters
        ----------
        y: float
            Real class.
        y_hat: float
            Output of the h function (i.e. the hypothesis of the logistic
             regression classifier.
         ----------
    
    Returns
        -------
        cost_i: float
            Value of the cost of the estimated output y_hat.
        -------
    """

    # ====================== YOUR CODE HERE ======================
    cost_i=y*np.log(y_hat)+(1-y)*np.log(1-y_hat)
    
    # ============================================================

    return cost_i


def fun_sigmoid(theta, x):
    """
    This function calculates the sigmoid function g(z), where z is a linear
    combination of the parameters theta and the feature vector X's components
    
    Parameters
        ----------
        theta: numpy vector
            Parameters of the h function of the logistic regression classifier.
        x: numpy vector
            Vector containing the data of one pattern.
         ----------
    
    Returns
        -------
        g: float
            Result of applying the sigmoid function using the linear
            combination of theta and X.
        -------
    """

    # ====================== YOUR CODE HERE ======================
    X_T=np.dot(theta,x.T)
    g=1/ (1+np.exp(-X_T))
    # ============================================================

    return g


def train_logistic_regression(X_train, Y_train, alpha):
    """
    This function implements the training of a logistic regression classifier
    using the training data (X_train) and its classes (Y_train).

    Parameters
        ----------
        X_train: Numpy array
            Matrix with dimensions (m x n) with the training data, where m is
            the number of training patterns (i.e. elements) and n is the number
            of features (i.e. the length of the feature vector which
            characterizes the object).
        Y_train: Numpy vector
            Vector that contains the classes of the training patterns. Its
            length is m.

    Returns
        -------
        theta: numpy vector
            Vector with length n (i.e, the same length as the number of
            features on each pattern). It contains the parameters theta of the
            hypothesis function obtained after the training.

    """
    # CONSTANTS
    # =================
    verbose = True
    max_iter = 500  # You can try with a different number of iterations
    # =================

    # Number of training patterns.
    m = np.shape(X_train)[0]

    # Allocate space for the outputs of the hypothesis function for each
    # training pattern
    h_train = np.zeros(shape=(1, m))

    # Allocate spaces for the values of the cost function on each iteration
    J = np.zeros(shape=(1, 1 + max_iter))

    # Initialize the vector to store the parameters of the hypothesis function
    theta = np.zeros(shape=(1,  1+np.shape(X_train)[1]))

    # -------------
    # CALCULATE THE VALUE OF THE COST FUNCTION FOR THE INITIAL THETAS
    # -------------
    # a. Intermediate result: Get the error for each element to sum it up.
    total_cost = 0
    for i in range(m):

        # Add a 1 (i.e., the value for x0) at the beginning of each pattern
        x_i = np.insert(np.array([X_train[i]]), 0, 1, axis=1)

        # Expected output (i.e. result of the sigmoid function) for i-th
        # pattern
        # ====================== YOUR CODE HERE ======================
        h_train[0,i]= fun_sigmoid(theta, x_i)
        # ============================================================

        # Calculate the cost for the i-the pattern and add it to the cost of
        # the last patterns
        # ====================== YOUR CODE HERE ======================
        total_cost = total_cost + calculate_cost_LogReg(Y_train[i], h_train[0,i])
        # ============================================================

    # b. Calculate the total cost
    # ====================== YOUR CODE HERE ======================
    total_cost=-1/m*total_cost
    J[0,0]=total_cost
    # ============================================================

    # -------------
    # GRADIENT DESCENT ALGORITHM TO UPDATE THE THETAS
    # -------------
    # Iterative method carried out during a maximum number (max_iter) of
    # iterations
    for num_iter in range(max_iter):

        # ------
        # STEP 1. Calculate the value of the h function with the current theta
        # values
        # FOR EACH SAMPLE OF THE TRAINING SET
        gradient=np.zeros(shape=(m, 1+ np.shape(X_train)[1]))     
        for i in range(m):
            # Add a 1 (i.e., the value for x0) at the beginning of each pattern
            x_i = np.insert(np.array([X_train[i]]), 0, 1, axis=1)
            # Expected output (i.e. result of the sigmoid function) for i-th
            # pattern
            # ====================== YOUR CODE HERE ======================
            h_train[0,i]= fun_sigmoid(theta, x_i)
            
            # ============================================================
            gradient[i,:]=(h_train[0,i]-Y_train[i])*x_i
            
            
        # ------
        # STEP 2. Update the theta values. To do it, follow the update
        # equations that you studied in the theoretical session
        # ====================== YOUR CODE HERE ======================
        theta=theta-alpha*(1/m)*sum(gradient)
        # ============================================================

        # ------
        # STEP 3: Calculate the cost on this iteration and store it on vector J
        # ====================== YOUR CODE HERE ======================
        total_cost=0
        for i in range(m):
            x_i = np.insert(np.array([X_train[i]]), 0, 1, axis=1)
            h_train[0,i]=fun_sigmoid(theta,x_i)
            total_cost=total_cost+calculate_cost_LogReg(Y_train[i],h_train[0,i])
        
        J[0,num_iter+1] = -1/m*total_cost
        # ============================================================

    # If verbose is True, plot the cost as a function of the iteration number
    if verbose:
        plt.plot(J[0],color='red')
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost J')
        plt.title('Cost function over the iterations with alpha=1e-3 et max_iter=500')
        plt.show()

    return theta


def classify_logistic_regression(X_test, theta):
    """
    This function returns the probability for each pattern of the test set to
    belong to the positive class using the logistic regression classifier.

    Parameters
        ----------
        X_test: Numpy array
            Matrix with dimension (m_t x n) with the test data, where m_t
            is the number of test patterns and n is the number of features
            (i.e. the length of the feature vector that define each element).
        theta: numpy vector
            Parameters of the h function of the logistic regression classifier.

    Returns
        -------
        y_hat: numpy vector
            Vector of length m_t with the estimations made for each test
            element by means of the logistic regression classifier. These
            estimations corredspond to the probabilities that these elements
            belong to the positive class.
    """

    num_elem_test = np.shape(X_test)[0]
    y_hat = np.zeros(shape=(1, num_elem_test))

    for i in range(num_elem_test):
        # Add a 1 (value for x0) at the beginning of each pattern
        x_test_i = np.insert(np.array([X_test[i]]), 0, 1, axis=1)
        # ====================== YOUR CODE HERE ======================

        y_hat[0, i] =fun_sigmoid(theta, x_test_i)
        # ============================================================

    return y_hat


# %%
# -------------
# MAIN PROGRAM
# -------------

dir_output = "Output"
features_path = dir_output + "/features_geometric.h5"
labels_path = dir_output + "/labels_high-low.h5"
test_size = 0.3

# -------------
# PRELIMINARY: LOAD DATASET AND PARTITION TRAIN-TEST SETS (NO NEED TO CHANGE
# ANYTHING)
# -------------

# import features and labels
h5f_data = h5py.File(features_path, 'r')
h5f_label = h5py.File(labels_path, 'r')

features_string = h5f_data['dataset_inserts_geometric']
labels_string = h5f_label['dataset_inserts_geometric']

X = np.array(features_string)
Y = np.array(labels_string)

h5f_data.close()
h5f_label.close()


# SPLIT DATA INTO TRAINING AND TEST SETS
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y,
                                                      test_size=test_size,
                                                      random_state=42)

# STANDARDIZE DATA
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print("Mean of the training set: {}".format(X_train.mean(axis=0)))
print("Std of the training set: {}".format(X_train.std(axis=0)))
print("Mean of the test set: {}".format(X_test.mean(axis=0)))
print("Std of the test set: {}".format(X_test.std(axis=0)))


# -------------
# PART 2.1: TRAINING OF THE CLASSIFIER AND CLASSIFICATION OF THE TEST SET
# -------------

# TRAINING

# Learning rate. Change it accordingly, depending on how the cost function
# evolve along the iterations
alpha = 1

# The function fTrain_LogisticReg implements the logistic regression
# classifier. Open it and complete the code.
theta = train_logistic_regression(X_train, Y_train, alpha)

print(theta)

# -------------
# CLASSIFICATION OF THE TEST SET
# -------------
Y_test_hat = classify_logistic_regression(X_test, theta)

# Assignation of the class: If the probability is higher than or equal
# to 0.5, then assign it to class 1
Y_test_asig = Y_test_hat >= 0.5

# -------------
# PART 2.2: PERFORMANCE OF THE CLASSIFIER: CALCULATION OF THE ACCURACY AND
# FSCORE
# -------------

# Show confusion matrix
Y_test= np.array([Y_test.astype(bool)])
confusion_matrix = metrics.confusion_matrix(Y_test.T,Y_test_asig.T )
print(confusion_matrix)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()


# -------------
# ACCURACY AND F-SCORE
# -------------
# ====================== YOUR CODE HERE ======================

Confusion_matrix=np.zeros((2,2))
for i in range(np.shape(Y_test_asig)[1]):
    if ((Y_test[0,i]==True).any() and (Y_test_asig[0,i]==True).any()):
        Confusion_matrix[0,0]=Confusion_matrix[0,0]+1
    if ((Y_test[0,i]==True).any() and (Y_test_asig[0,i]==False).any()):
        Confusion_matrix[0,1]=Confusion_matrix[0,1]+1
    if ((Y_test[0,i]==False).any() and (Y_test_asig[0,i]==True).any()):
        Confusion_matrix[1,0]=Confusion_matrix[1,0]+1
    if ((Y_test[0,i]==False).any() and (Y_test_asig[0,i]==False).any()):
        Confusion_matrix[1,1]=Confusion_matrix[1,1]+1

print(Confusion_matrix)
accuracy=Confusion_matrix.trace()/(Confusion_matrix[0,0]+Confusion_matrix[1,0]+Confusion_matrix[0,1]+Confusion_matrix[1,1])
Precision=Confusion_matrix[0,0]/(Confusion_matrix[0,0]+Confusion_matrix[1,0])
Recall=Confusion_matrix[0,0]/(Confusion_matrix[0,0]+Confusion_matrix[0,1])
f_score=2*((Precision*Recall)/(Precision+Recall))
# ============================================================



print("***************")
print("The accuracy of the Logistic Regression classifier is {:.4f}".
      format(accuracy))
print("***************")




print("***************")
print("The Recall of the Logistic Regression classifier is {:.4f}".
      format(Recall))
print("***************")


print("")
print("***************")
print("The Precision  of the Logistic Regression classifier is {:.4f}".
      format(Precision))
print("***************")



print("")
print("***************")
print("The F1-score of the Logistic Regression classifier is {:.4f}".
      format(f_score))
print("***************")

#------------------
# ROC curve
# ----------------

fpr, tpr, threshold = metrics.roc_curve(Y_test.T, Y_test_asig.T)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Logistic Regression ROC curve ')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


#--------------
# Other classifier
#----------------




#-----------------
#SVM
#-------------------
ModelSVM = svm.SVC(kernel='rbf')
ModelSVM.fit(X_train, Y_train)
#Prediction
PredSVM=ModelSVM.predict(X_test)
Y_test1 = Y_test.transpose()
Accuracy=accuracy_score(Y_test1, PredSVM)
F1_score=f1_score(y_true=Y_test1, y_pred=PredSVM)
print("***************")
print("The accuracy of the SVM classifier is {:.4f}".
      format(Accuracy))
print("***************")


print("")
print("***************")
print("The F1-score of the SVM classifier is {:.4f}".
      format(F1_score))
print("***************")
#Confusion Matrix
confusion_matrix = metrics.confusion_matrix(Y_test1,PredSVM )
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()
#ROC curve
metrics.plot_roc_curve(ModelSVM, X_test, Y_test1) 
plt.show()



#-----------------
#Decison Tree
#---------------
ModelTree=DecisionTreeClassifier(criterion='entropy',max_depth=3)
ModelTree.fit(X_train,Y_train)
#Prediction
PredTree=ModelTree.predict(X_test)
plot_tree(ModelTree,filled=True)
plt.show()
Accuracy=accuracy_score(Y_test1, PredTree)
F1_score=f1_score(y_true=Y_test1, y_pred=PredTree)
print("***************")
print("The accuracy of the Decsion Tree classifier is {:.4f}".
      format(Accuracy))
print("***************")


print("")
print("***************")
print("The F1-score of the Decision Tree classifier is {:.4f}".
      format(F1_score))
print("***************")
#Confusion Matrix
confusion_matrix = metrics.confusion_matrix(Y_test1, PredTree)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

#ROC curve
metrics.plot_roc_curve(ModelTree, X_test, Y_test1) 
plt.show()


#-------------------------
# Random Forest
#-----------------------
ModelRanF=RandomForestClassifier(n_estimators=100,criterion='entropy')
ModelRanF.fit(X_train,Y_train)
#Prediction
PredRanF=ModelRanF.predict(X_test)
Accuracy=accuracy_score(Y_test1, PredRanF)
F1_score=f1_score(y_true=Y_test1, y_pred=PredRanF)
print("***************")
print("The accuracy of the Random Forest classifier is {:.4f}".
      format(Accuracy))
print("***************")


print("")
print("***************")
print("The F1-score of the Random Forest classifier is {:.4f}".
      format(F1_score))
print("***************")

#Confusion Matrix
confusion_matrix = metrics.confusion_matrix(Y_test1, PredRanF)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

#Roc curve
metrics.plot_roc_curve(ModelRanF, X_test, Y_test1) 
plt.show()


#-------------------
#RNN
#--------------------
y_train=to_categorical(Y_train)
y_test=to_categorical(Y_test1)
model=Sequential()
n_cols=X.shape[1]
model.add(Dense(5,activation="relu",input_shape=(n_cols,)))
model.add(Dense(5,activation="relu"))
model.add(Dense(5,activation="relu"))
model.add(Dense(5,activation="relu"))
model.add(Dense(5,activation="relu"))
model.add(Dense(2,activation="sigmoid"))
model.compile(optimizer="adam",loss='binary_crossentropy',metrics=["accuracy"])
model.fit(X_train,y_train,epochs=10)
#Prediction
predRNN=model.predict(X_test)
Accurracy=model.evaluate(X_test,y_test,verbose=0)
print("")
print("***************")
print("Accuracy of RNN is:{} \n Error  of RNN is:{}".
      format(Accurracy[1],1-Accurracy[1]))


#------------------
# Logistic Regression
#-------------------
ModelLogr=LogisticRegression()
ModelLogr.fit(X_train,Y_train)
#Prediction
PredLogr=ModelLogr.predict(X_test)
Accuracy=accuracy_score(Y_test1, PredTree)
F1_score=f1_score(y_true=Y_test1, y_pred=PredLogr)
print("***************")
print("The accuracy of Logistic Regression is {:.4f}".
      format(Accuracy))
print("***************")
print("")
print("***************")
print("The F1-score of Logistic Regression is {:.4f}".
      format(F1_score))
print("***************")
#Confusion Matrix
confusion_matrix = metrics.confusion_matrix(Y_test1, PredLogr)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()
#ROC curve
metrics.plot_roc_curve(ModelLogr, X_test, Y_test1) 
plt.show()

# ----------
# KNN
# ---------------
errors = []
for k in range(1,9):
    knn = KNeighborsClassifier(n_neighbors=k)
    errors.append(100*(1 - knn.fit(X_train, Y_train).score(X_test, Y_test1)))
plt.plot(range(1,9), errors, 'o-')
plt.show()
Modelknn = KNeighborsClassifier(n_neighbors = 4)
Modelknn.fit(X_train,Y_train)
#Prediction
PredKnn=Modelknn.predict(X_test)
Accuracy=accuracy_score(Y_test1,PredKnn)
F1_score=f1_score(y_true=Y_test1, y_pred=PredKnn)
print("***************")
print("The accuracy of KNN is {:.4f}".
      format(Accuracy))
print("***************")
print("")
print("***************")
print("The F1-score of KNN is {:.4f}".
      format(F1_score))
print("***************")
#Confusion Matrix
confusion_matrix = metrics.confusion_matrix(Y_test1, PredKnn)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()
#ROC curve
metrics.plot_roc_curve(Modelknn, X_test, Y_test1) 
plt.show()



# Comparison between the different methods
#Accuracy and F1_score
values = [0.8852,0.8689,0.8033,0.8033,0.8361,0.8689,0.770]
val=[0.6957,0.600,0.5385,0.6667,0.5833,0.600,0]
mydata = pd.DataFrame({"Accuracy":values,"F1_score":val})
mydata.index = ['Our Model','SVM', 'Tree', 'LogRegr','Forest','KNN','RNN']
mydata.plot(kind="bar",rot=30) 
plt.show()

# confusion matrix 
fig, axs = plt.subplots(1, 6, figsize = (25, 5))
Pred=[Y_test_asig,PredSVM,PredTree,PredLogr,PredRanF,PredKnn]
names= [ "Our model","SVM", "Decision Tree","Logistic Regression", "Random Forest", "kNN"]

for i, pred in enumerate(Pred):
    confusion_matrix = metrics.confusion_matrix(Y_test.T,pred.T )
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot(ax = axs[i])
    axs[i].set_title(names[i], fontsize = 15)
#ROC curve   
fig, axs = plt.subplots(1, 5, figsize = (25, 5))   
Model=[ModelSVM,ModelTree,ModelLogr,ModelRanF,Modelknn]
names= [ "SVM", "Decision Tree","Logistic Regression", "Random Forest", "kNN"]
for i, model in enumerate(Model):
    metrics.plot_roc_curve(model, X_test, Y_test.T,ax=axs[i]) 
    axs[i].set_title(names[i], fontsize = 15)


