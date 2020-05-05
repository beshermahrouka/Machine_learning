import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix 




def fit_perceptron(X_train, y_train):
   
    #create and initilize "w" matrix with zeros values
    w = np.zeros((1,np.size(X_train, 1) + 1))
      
    #create and initilize "w_best" matrix with zeros values, used for update "w"
    w_best = np.zeros((1,np.size(X_train, 1) + 1))
    
    #repeat the algorithm "n" time for pocket algorithm 
    repetition = 1000
    
    #add extra column with one values
    my_ones = np.ones((np.size(X_train, 0),1))
    X_train = np.hstack((X_train,my_ones))
     
    #initilize the error to zero before update
    error_w_new = 0
    error_w_best = 0
    
    #get the total number of data points
    num_rows = np.size(X_train, 0) 
    
    #loop to repeat the pocket algorithm "n" time
    for j in range(0, repetition):
        
        #index for accessing data_point from data set
        i = 1  
          
        #loop throgh x_train matrix to get different data point
        while i  <= num_rows:
          
           # "i" data point 
           data_point = X_train[i-1:i]
          
           # "i" y_value
           y_value = y_train[i-1:i]
         
           #convert to a vector  
           data_point = data_point.flatten()
          
           #convert to a vector
           y_value = y_value.flatten()
          
           #convert y_value to a number
           y_value = y_value[0]
          
           #convert "w" as a vector
           w = w.flatten()  
          
           #convert "w_best" as a vector
           w_best = w_best.flatten()
         
           #get the predicted value of y based on "w"  
           predicted_value = pred(data_point,w)
          
           #if predicted value different from y_value then update w and get error 
           if predicted_value != y_value:
            
               #update algorithm for "w" vector
               w = w + y_value * data_point
                 
               #get error average for based on "w_best" vector
               error_w_best = errorPer(X_train,y_train,w_best)
            
               #get error average for based on "w_new" vector
               error_w_new = errorPer(X_train,y_train,w)
            
               #if average error for the updated "w" less than "w_best" then make "w_best" equal to updated "w"
               if error_w_new <= error_w_best:
                              
                 w_best = w     
                 
               break
     
           i = i + 1
    #return the best weight vector "w"     
    return w_best       
def errorPer(X_train,y_train,w):
     
    # intilized my_sum to zero
    my_sum = 0
    
    #index to get different data point 
    i = 1
    
    #number of rows
    num_rows = np.size(X_train, 0) 
    
    #loop to iterate through different data point
    while i  <= num_rows:
        
      #get an "i" data point   
      data_point = X_train[i-1:i,:]
      
      #convert data point to a vector
      data_point = data_point.flatten()
     
      #get the y value for the data point  
      y_value = y_train[i-1:i]
     
      #convert y value to a vector  
      y_value = y_value.flatten()
     
      #get the y value as a number  
      y_value = y_value[0]
     
      #get the predicted y value based on "w"  
      predicted_value = pred(data_point,w)
     
      # if y value id different than predicted value then add one to my_sum 
      if predicted_value != y_value:
         
         my_sum = my_sum + 1      
       
      i = i + 1
    
    #get average error by dividing total sum by total number of data points
    avgError = my_sum / num_rows 
    
    #return avergae error
    return avgError 
def confMatrix(X_train,y_train,w):
    
    #initialize the confusion matrix to zeros
    my_confmatrix = np.array([[0,0],[0,0]])
    
    #index to access different data point
    i = 1
    
    #get total number of rows which is the total number of data points
    num_rows = np.size(X_train, 0)
    
    #add new column to x_training matrix with one values
    my_ones = np.ones((np.size(X_train, 0),1)) 
    X_train = np.hstack((X_train,my_ones))
    
    #loop to iterate through different data point
    while i  <= num_rows:
      
     #get an "i" data point   
     data_point = X_train[i-1:i,:]
     
     #convert data point to a vector
     data_point = data_point.flatten()
     
     #get the y value for the data point
     y_value = y_train[i-1:i]
     
     #convert y value to a vector
     y_value = y_value.flatten()
     
     #get the y value as a number
     y_value = y_value[0]
     
     #get the predicted value of y based on "w"
     predicted_value = pred(data_point,w)
     
     #count of the total number of points from the training set that have been correctly classified to be class −1 by the linear classifier defined by w
     if predicted_value == -1 and y_value == -1:
         
       my_confmatrix[0][0] = my_confmatrix[0][0] + 1
      
     #count of the total number of points from the training set that are in class 1 but are classified by w to be in class −1   
     elif predicted_value == -1 and y_value == 1: 
         
       my_confmatrix[0][1] = my_confmatrix[0][1] + 1
       
     #count of the total number of points from the training set that have been correctly classified to be class 1 by the linear classifier defined by w  
     elif predicted_value == 1 and y_value == -1:
         
       my_confmatrix[1][0] = my_confmatrix[1][0] + 1
       
     #count of the total number of points from the training set that are in class −1 but are classified by w to be in class 1  
     elif predicted_value == 1 and y_value == 1: 
         
       my_confmatrix[1][1] = my_confmatrix[1][1] + 1    
       
     i = i + 1
    
    #return the confusion matrix
    return my_confmatrix      
def pred(X_train,w):
    
    #convert X_train to a vector
    X_train = X_train.flatten()
   
    #convert "w" to a vector
    w = w.flatten()
    
    #transpose "w" vector
    w_transpose = np.transpose(w)
    
    #get the dot product for x_train and w 
    dot_product = np.dot(w_transpose,X_train)
    
    # if the dot product result strictly positive then it belongs to class 1 otherwise class -1
    if dot_product > 0:
        
        return 1
    
    elif dot_product <= 0:
        
        return -1 
def test_SciKit(X_train, X_test, Y_train, Y_test):

    #create perceptron object
    per = Perceptron()
    
    #train the model
    per.fit(X_train,Y_train)
    
    #get predictions using testing sets
    predictions = per.predict(X_test)
    
    #return confusion matrix for this model
    return confusion_matrix(Y_test,predictions)
    
def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2)
    
    for i in range(80):
        if y_train[i]==1:
            y_train[i]=-1
        else:
            y_train[i]=1
    for j in range(20):
        if y_test[j]==1:
            y_test[j]=-1
        else:
            y_test[j]=1
            
    #Testing Part 1a
    w=fit_perceptron(X_train,y_train)
    cM=confMatrix(X_test,y_test,w)
    
    #Testing Part 1b
    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)
    

test_Part1()
