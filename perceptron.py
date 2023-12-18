# Ogochukwu Jane Okafor

# PERCEPTRON ALGORITHM

import pandas as pd
import numpy as np

# Assigning trainData to represent our train data saved as a csv file.
trainData = pd.read_csv('/train.data', header = None)

# Filtering out the data in the train data set into two classes separately
filtered_trainData_1and2 = trainData[(trainData.iloc[:, -1] == 'class-1') | (trainData.iloc[:, -1] == 'class-2')] # Filtering out only class-1 and class-2 from the train data
filtered_trainData_2and3 = trainData[(trainData.iloc[:, -1] == 'class-2') | (trainData.iloc[:, -1] == 'class-3')] # Filtering out only class-2 and class-3 from the train data
filtered_trainData_1and3 = trainData[(trainData.iloc[:, -1] == 'class-1') | (trainData.iloc[:, -1] == 'class-3')] # Filtering out only class-1 and class-3 from the train data

# Selecting the features X and class labels y of class 1 and class 2 for the train data
D_of_1and2 = np.array(filtered_trainData_1and2)
D_of_1and2 [:,-1][D_of_1and2 [:,-1] == 'class-1'] = 1 # labelling class 1 to 1 
D_of_1and2 [:,-1][D_of_1and2 [:,-1] == 'class-2'] = -1 # labelling class 2 to -1

# Selecting the features X and class labels y of class 2 and class 3 for the train data
D_of_2and3 = np.array(filtered_trainData_2and3)
D_of_2and3 [:,-1][D_of_2and3 [:,-1] == 'class-2'] = 1 # labelling class 2 to 1 
D_of_2and3 [:,-1][D_of_2and3 [:,-1] == 'class-3'] = -1 # labelling class 3 to -1 

# Selecting the features X and class labels y of class 1 and class 3 for the train data
D_of_1and3 = np.array(filtered_trainData_1and3)
D_of_1and3 [:,-1][D_of_1and3 [:,-1] == 'class-1'] = 1 # labelling class 1 to 1 
D_of_1and3 [:,-1][D_of_1and3 [:,-1] == 'class-3'] = -1 # labelling class 3 to -1 

# Getting a perceptron train function to train our data
def PerceptronTrain(D, MaxIter):
  """ 
      Perceptron Train Function: 
      This function represents the Perceptron algorithm for the train dataset.
  
      Parameters:
            1. Train Data (D): 
                  Arguments (ndarray): 
                - Indicates a numpy array containing the train data in which every row of this array is a train data.
                - The last object of each row is the class label.
                - Excluding the last column in each row, every other row identify as features(X).
            
            2. MaxIter:
                - Int: The maximum number of iterations for the Perceptron train data, which is 20.
           
      Returns:
                - A tuple containing the final bias (b) and the final weight (W) for train data for class 1 and 2,
                  class 2 and class 3 and class 1 and class 3.
                - It also returns a tuple containing the final bias and the final weight (W) for train data under one vs rest; 
                  that is it returns that for class 1 vs rest, for class 2 vs rest, and for class 3 vs rest.
      
  """
  Weights = np.zeros((4,1)) # Equating the weights to zero for a 4 x 1 matrix.
  b = 0 # bias is equal to zero (0)
  for iter in range(MaxIter):  # for each number of iterations (1,2,...,20).
    for obj in D:# for every object in the training dataset.
      X = np.reshape(obj[:4],(4,1)) # the shape of X vector (feature) is a 4 x 1 matrix.
      y = obj[-1] # y represents the class labels where by -1 means the class column in each row.
    
      # setting a to be the activation score
      a = np.dot(Weights.transpose(),X) + b # taking the transpose of the weights, feature plus bias equals a.
      if y * a <= 0: #  if the class label multiplied by the activation score is less than or equal to zero (0).
        Weights = Weights + y * X # new weights are derived from previous weights + class label x feature.
        b += y # bias + or equal to class label
  return b, Weights # final bias and weights are return
# PerceptronTrain(D_of_1and2,20)
# PerceptronTrain(D_of_2and3,20)
# PerceptronTrain(D_of_1and3,20)

# Assigning testData to represent our test data saved as a csv file.
testData = pd.read_csv('/test.data', header = None)

# Filtering all classes in two classes separately for the test data sets
filtered_testData_1and2 = testData[(testData.iloc[:, -1] == 'class-1') | (testData.iloc[:, -1] == 'class-2')] # Filtering out only class-1 and class-2 from the test data.
filtered_testData_2and3 = testData[(testData.iloc[:, -1] == 'class-2') | (testData.iloc[:, -1] == 'class-3')] # Filtering out only class-2 and class-3 from the test data.
filtered_testData_1and3 = testData[(testData.iloc[:, -1] == 'class-1') | (testData.iloc[:, -1] == 'class-3')] # Filtering out only class-1 and class-3 from the test data.

# Selecting the features X and class labels y of class 1 and class 2 for the test data
Dtest_1and2 = np.array(filtered_testData_1and2)
Dtest_1and2 [:,-1][Dtest_1and2 [:,-1] == 'class-1'] = 1 # labelling class 1 to 1. 
Dtest_1and2 [:,-1][Dtest_1and2 [:,-1] == 'class-2'] = -1 # labelling class 2 to -1. 

# Selecting the features X and class labels y of class 2 and class 3 for the test data
Dtest_2and3 = np.array(filtered_testData_2and3)
Dtest_2and3 [:,-1][Dtest_2and3 [:,-1] == 'class-2'] = 1 # labelling class 2 to 1. 
Dtest_2and3 [:,-1][Dtest_2and3 [:,-1] == 'class-3'] = -1 # labelling class 3 to -. 

# Selecting the features X and class labels y of class 1 and class 3 for the test data
Dtest_1and3 = np.array(filtered_testData_1and3)
Dtest_1and3 [:,-1][Dtest_1and3 [:,-1] == 'class-1'] = 1 # labelling class 1 to 1. 
Dtest_1and3 [:,-1][Dtest_1and3 [:,-1] == 'class-3'] = -1 # labelling class 3 to -1. 

# getting the biased weights of the classes
b_Weights_1and2 = PerceptronTrain(D_of_1and2, 20) # biased weights for classes 1 and 2.
b_Weights_2and3 = PerceptronTrain(D_of_2and3, 20) # biased weights for classes 2 and 3.
b_Weights_1and3 = PerceptronTrain(D_of_1and3, 20) # biased weights for classes 1 and 3.

# getting the X for the classes in the train data for binary classification
#X_1and2 = np.reshape(D_of_1and2[0][0:4],(4,1)) 
#X_2and3 = np.reshape(D_of_2and3[0][0:4],(4,1)) 
#X_1and3 = np.reshape(D_of_1and3[0][0:4],(4,1)) 

# Defining a perceptron test function to test our trained data
def PerceptronTest(BW, X):
  """
      Perceptron Test Function:
        This function determines an perceptron's output bias, weight and vector X. 
        This function tests the the binary classes derived from the train data.

      Parameters:
          1. BW:
              - A tuple containing the bias and weight vector of the perceptron.
                
          2. Array; 
              - Represents the given vector(s) to be evaluated by the perceptron.

      Returns:
              - Numpy Array
              - Signs showing how the perceptron performed on each component of the input vector from the train data.
  """
  b = BW[0] # first index of BW is the bias which is index 0.
  Weights = BW[1] # second index of BW are the Weights which is 2.
  # Setting our activation score for the perceptron train.
  a = np.dot(Weights.transpose(), X) + b #taking the transpose of the weights, feature plus bias equals a.
  return np.sign(a) # returns the sign of our activation score 

# PerceptronTest(b_Weights_1and2, X_1and2) 
# PerceptronTest(b_Weights_2and3, X_2and3) 
# PerceptronTest(b_Weights_1and3, X_1and3) 

 # Defining the one versus rest function for multi-class classification
def OneVsRest(data, clone_class):
  """
      One Vs Rest Function:
            This functions converts a problem of binary classifier into to that of one-vs-rest multiclass classification.
            It aims to select a specific class amd assign a positive value to it while the remaining class to a negative class.

      Parameters:
          1. data: 
                - Nump Array
                - The data to transform. 
                - The last index is assumed to be the target class.
          2. clone_class : 
                - Any class
                - The targeted class will take positive one(1) while the remaining class will take negative (-1)

      Returns:
          - Numpy Array
          - A duplicate of the original data with the selected variable converted into a binary variable.
          - A targeted class is assigned a value of 1 while the remaining classes are assigned a value of -1.
  """
  # Duplicating the data to avoid modification
  new_data = np.copy(data)
  # Creating a boolean mask for any of the targeted class
  clones = new_data[:, -1] == clone_class
    # Designates any targeted class as 1 and the others as -1
  new_data[:, -1] = np.where(clones, 1, -1)
  return new_data # the new data as specified above is returned.

tr_class1 = OneVsRest(trainData, "class-1") # represents class 1 vs rest for the train data.
tr_class2 = OneVsRest(trainData, "class-2") # represents class 2 vs rest for the train data.
tr_class3 = OneVsRest(trainData, "class-3") # represents class 3 vs rest for the train data.

te_class1 = OneVsRest(testData, "class-1") # represents class 1 vs rest for the test data.
te_class2 = OneVsRest(testData, "class-2") # represents class 2 vs rest for the test data.
te_class3 = OneVsRest(testData, "class-3") # represents class 3 vs rest for the test data.
#PerceptronTrain(tr_class1, 20)
#PerceptronTrain(tr_class2, 20)
#PerceptronTrain(tr_class3, 20)

def PredictClass(b_weights, the_classes):
  """
      Predict Function: 
            This function predicts the class of each observation in the_classes based on the weights of a perceptron.

      Parameters:
            1. b_weights:
                     - A numpy array.
                     - The biased weight vector of for the train datasets for binary classification.
                    - A 2 dimensional numpy array.
                    - Enclosing the total number of observations and the total number of features.
                    - The first 'n' features columns contain the feature values.
                    - The last column contains the class labels.
      Returns:
            - predicts:
                  - A numpy array
                  - A 1-dimensional array for the length of total observations containing the predicted class for each observation in the classes.
                  - Returns the three classifiers for both perceptron train and test datasets in binary classification.
    """
  pred_list = [] # an empty list to hold the predictions
  for m in range(the_classes.shape[0]): # for every class
      X = np.reshape(the_classes[m][0:4],(4,1)) # the shape of X vector (feature) is a 4 x 1 matrix.
      p = PerceptronTest(b_weights, X) # Let be serve as the perceptron test's biased-weights and class features X.
      pred_list.append(p[0][0]) # appending each predict to the predict list.
      predicts = np.array(pred_list) # assigning 'predicts' to be the new name of the prediction list in the form of numpy array.
  return predicts # the prediction is returned

# predictions for all the class binary combinations for the train data sets
predicted_values_1and2 = PredictClass(b_Weights_1and2, D_of_1and2) # predicted values for class 1 and class 2 for the perceptron train data.
predicted_values_2and3 = PredictClass(b_Weights_2and3, D_of_2and3) # predicted values for class 2 and class 3 for the perceptron train data.
predicted_values_1and3 = PredictClass(b_Weights_1and3, D_of_1and3) # predicted values for class 1 and class 3 for the perceptron train data.

# predictions for all the class binary combinations for the test data sets
predicted_test_1and2 = PredictClass(b_Weights_1and2, Dtest_1and2) # predicted values for class 1 and class 2 for the perceptron test data.
predicted_test_2and3 = PredictClass(b_Weights_2and3, Dtest_2and3) # predicted values for class 2 and class 3 for the perceptron test data.
predicted_test_1and3 = PredictClass(b_Weights_1and3, Dtest_1and3) # predicted values for class 1 and class 3 for the perceptron test data.

def PredictMultiClass(pmc1, pmc2, pmc3, X_feature):
    """
    Predict Multi Class Function 
              This functionpredicts class label of a dataset under one versus rest method of multi-class classification.
    
    Parameters:
        1. pmc1: 
                  - A tuple consisting the bias and weights for class 1
        2. pmc2      
                  - A tuple consisting the bias and weights for class 2
        3. pmc3   
                  - A tuple consisting the bias and weights for class 3
        4. X_feature:   
                  - Numpy array
                  - Dataset consisting the number of features 
    Returns:
        new_y:   
                  - Numpy array
                  - Consists the predicted class labels ordered according to records in X_feature.
    """
    # Designating an array to store class labels predictions
    new_y = np.empty(X_feature.shape[0],  dtype=int)
    # Iterating over every feature in X_feature
    for f in range(X_feature.shape[0]):
      X = X_feature[f,:4].reshape(4, 1)
      # Get the activation scores for each class
      first_score = np.dot(pmc1[1].T, X) + pmc1[0]
      second_score = np.dot(pmc2[1].T, X) + pmc2[0]
      third_score = np.dot(pmc3[1].T, X) + pmc3[0]
      data_scores = np.array([first_score, second_score, third_score])
      data_point = np.argmax(data_scores) 
      new_y[f] = data_point + 1
    return  new_y

new_train = np.array(trainData.iloc[:, 0:5])
train_maps   = {'class-1' : 1, 'class-2' : 2, 'class-3' : 3}
y_trains = trainData.iloc[:, -1]
d_classes = np.array([train_maps[map] for map in y_trains])

 # Horizontally joining the X and y together for the train data
X_col = d_classes.reshape(-1,1)
X_feat = trainData.iloc[:, :-1]
d_mul_train = np.hstack((X_feat, X_col))

# getting the biased weights of the classes for multi class classificiation
bW1 = PerceptronTrain(tr_class1, 20) # biased weights for class 1 vs rest for multi-class classification.
bW2 = PerceptronTrain(tr_class2, 20) # biased weights for class 2 vs rest for multi-class classification.
bW3 = PerceptronTrain(tr_class3, 20) # biased weights for class 3 vs rest for multi-class classification.

# Computing for test data
new_test = np.array(testData.iloc[:, 0:5])
test_maps  = {'class-1' : 1, 'class-2' : 2, 'class-3' : 3}
y_test = testData.iloc[:, -1]
t_class = np.array([test_maps[map] for map in y_test])

# Horizontally joining the X and y together for the test data
Xt_col = t_class.reshape(-1,1)
y_tests = testData.iloc[:, :-1]
d_mul_test = np.hstack((y_tests, Xt_col))

# Computing predicted values for both train and test data sets
train_predictions = PredictMultiClass((bW1), (bW2), (bW3), new_train)
test_predictions = PredictMultiClass((bW1), (bW2), (bW3), new_test)

def ComputeAccuracy(pred, true):
  """
      Accuracy Function:
            This function computes accuracies for the perceptron train data and perceptron test data for the binary classifiers.
      Parameters:
            1. pred: 
                  - Numpy Array  for any predicted class.
                  - Represnting positive class with 1 and -1 representing negative class.
            2. true:
                  - Numpy Array  for any predicted class.
                  - Represnting positive class with 1 and -1 representing negative class.
      Returns:
            A float:
                 - Accuracy of the three binary classifiers in perceptron train and test datasets.       
  """
  positive_mask = true == 1 # gives a positive mask if true
  positive = np.count_nonzero(positive_mask) # counts the number of elements in the positive class 
  tp = np.count_nonzero(pred[positive_mask]==1) # counts true positive
  fn = np.count_nonzero(pred[positive_mask]==-1) # counts false negative (fn)
  negative_mask = true == -1 # gives  a negative mask if true
  negative = np.count_nonzero(negative_mask) # countsthe number of elements in the negative class 
  fp = np.count_nonzero(pred[negative_mask]==1) # counts false positive (fp)
  tn = np.count_nonzero(pred[negative_mask]==-1) # counts true negative (tn)
  accuracy = (tp + tn)/(tp + tn + fp + fn) # (tp + tn)/(tp + tn + fp + fn) determines accuracy.
  return accuracy # returns all accuracies for both train and test data, for perceptron, one vs rest, and regularisation.

# true values of the train datasets
true_values_1and2 = D_of_1and2[:, -1] # true values of class 1 and class 2 for the train data.
true_values_2and3 = D_of_2and3[:, -1] # true values of class 2 and class 3 for the train data.
true_values_1and3 = D_of_1and3[:, -1] # true values of class 1 and class 3 for the train data.
  
  # true values of the test datasets
true_test_1and2 = Dtest_1and2[:, -1] # true values of class 1 and class 2 for the test data.
true_test_2and3 = Dtest_2and3[:, -1] # true values of class 2 and class 3 for the test data.
true_test_1and3 = Dtest_1and3[:, -1] # true values of class 1 and class 3 for the test data.

# calling the accuracies for all class combinations for the train datasets
ac1_2 = ComputeAccuracy(predicted_values_1and2, D_of_1and2[:, -1]) # accurate value for perceptron train data class 1 and 2.
ac2_3 = ComputeAccuracy(predicted_values_2and3, D_of_2and3[:, -1]) # accurate value for perceptron train data class 2 and 3.
ac1_3 = ComputeAccuracy(predicted_values_1and3, D_of_1and3[:, -1]) # accurate value for perceptron train data class 1 and 3.
print("Accuracy for class 1 and class 2 for the perceptron train after 20 iterations = ", ac1_2)
print("Accuracy for class 2 and class 3 for the perceptron train after 20 iterations = ", ac2_3)
print("Accuracy for class 1 and class 3 for the perceptron train after 20 iterations = ", ac1_3)

# calling the accuracies for all class combinations for the test datasets
act1_2 = ComputeAccuracy(predicted_test_1and2, Dtest_1and2[:, -1]) # accurate value for perceptron test data class 1 and 2.
act2_3 = ComputeAccuracy(predicted_test_2and3, Dtest_2and3[:, -1]) # accurate value for perceptron test data class 2 and 3.
act1_3 = ComputeAccuracy(predicted_test_1and3, Dtest_1and3[:, -1]) # accurate value for perceptron test data class 1 and 3.
print("Accuracy for class 1 and class 2 for the perceptron test after 20 iterations = ", act1_2)
print("Accuracy for class 2 and class 3 for the perceptron test after 20 iterations = ", act2_3)
print("Accuracy for class 1 and class 3 for the perceptron test after 20 iterations = ", act1_3)


def MultiClassAccuracy(data, predicts):
  """
      Accuracy Function:
            This function computes accuracy for both train and test data for L2 regularisation.

      Parameters:
            1. data: 
                  - Numpy Array  for any data class.

            2. predicts:
                  - Numpy Array  for any predicted class.
      Returns:
            A float
                 - Accuracy of the multi-class classifiers in L2 Regularisation.
  """
  true_classes = data[:, -1] # true values of either train data or test data
  train_accuracy = np.sum(true_classes == predicts) / len(true_classes) # determining accuracy of classes
  return train_accuracy # returns training accuracy

OneVersusRest_Acc1 = print('Train Accuracy:', MultiClassAccuracy(d_mul_train, train_predictions)) # computing accuracy for the train data
OneVersusRest_Acc2 = print('Test Accuracy:', MultiClassAccuracy(d_mul_test, test_predictions)) # computing accuracy for the test data
  
# Defining a function for L2 Regularisation
def L2Regularisation(D, MaxIter, co_of_reg):
  """
      L2 Regularisation Function: 
            Thiss function implements the L2 regularisation to the given datasets.

      Parameters:
            1. Train Data (D):
                  - Numpy Array.
                  - Contains all class features and labels.
            2. MaxIter:
                  - Int.
                  - The maximum number of iterations (20) required to run the algorithm.
            3. co_of_reg:
                  - A float.
                  - Determines the strength of the regularisation.

      Returns:
                  - A tuple containing the bias and weights after implementing the L2 regularization
  """
  # initialize weights and bias to zero
  Weights = np.zeros((4,1)) # Equating the weights to zero for a 4 x 1 matrix.
  b = 0 # bias is equal to zero (0)
  for iter in range(MaxIter): # for all numbers in the range of iteration (1,2,...,20)
    for obj in D: # for every oject existing in the train data
      X = np.reshape(obj[:4],(4,1)) # the shape of X vector (feature) is a 4 x 1 matrix.
      y = obj[-1] # y represents the class labels where by -1 means the class column in each row.
      # setting a to be the activation score
      a = np.dot(Weights.transpose(),X) + b # taking the transpose of the weights, feature plus bias equals a.
      if y * a <= 0: #  if the class label multiplied by the activation score is less than or equal to zero (0).
        Weights = (1 - 2 * co_of_reg) *  Weights + y * X # 1 minus 2 multiply by coefficient of regularisation and weights + class label and X features determines weights.
      else:
        Weights = (1 - 2 * co_of_reg) * Weights # 1 minus 2 multiply by coefficient of regularisation and weights determines weights.
        b += y # previous bias (b) plus class label (y) determines the new bias
    return b, Weights # the final bias and final weight will be returned

# An enclosed list of the coefficients of regularisation 
coefficients = [0.01, 0.1, 1.0, 10.0, 100.0]

for co in coefficients: # for each coefficient existing in the list of regularisation coefficient
  # Computing biases and weights for all the classes in one vs rest under regularisation
  b_weight1 = L2Regularisation(tr_class1, 20, co) # biased weights for class 1 vs rest for L2 regularisation.
  b_weight2 = L2Regularisation(tr_class2, 20, co) # biased weights for class 2 vs rest for L2regularisation.
  b_weight3 = L2Regularisation(tr_class3, 20, co) # biased weights for class 3 vs rest for L2 regularisation.
  
  # Predictions for the train and test data in one vs rest under regularisation
  train_predictions = PredictMultiClass((b_weight1), (b_weight2), (b_weight3), new_train) # Predictions for the train data
  test_predictions = PredictMultiClass((b_weight1), (b_weight2), (b_weight3), new_test) # Predictions for the test data

  # computing accuracies for train and test data sets in one vs rest under regularisation
  OneVersusRest_Acc1 = print('The Train Accuracy:', MultiClassAccuracy(d_mul_train, train_predictions)) # prints accurcay for the train data
  OneVersusRest_Acc2 = print('The Test Accuracy:', MultiClassAccuracy(d_mul_test, test_predictions)) # prints accurcay for the test data
