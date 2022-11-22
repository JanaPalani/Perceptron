import random 
import numpy as np
class Perceptron:
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate 
    
    def generate_x(self,data_x):
        self.data_x = np.c_[np.ones((len(data_x),1)),data_x]
        return self.data_x
 
    
    def train(self,data_x,data_y,iteration):
        self.data_x = self.generate_x(data_x)
        self.actual = np.array(data_y)
        self.actual = np.reshape(self.actual,(len(self.actual),1))
        self.size = len(self.data_x[0])
        self.weights =np.random.randn((self.size),1)
        n = len(data_x)
        for i in range(iteration): 
            gradients = 1/n* self.data_x.T.dot(self.data_x.dot(self.weights) - self.actual)
            self.weights = self.weights - self.learning_rate*gradients
        return self.weights 
    
    def predict(self,test_x):
        self.test_x= self.generate_x(test_x) 
        return  self.test_x.dot(self.weights)
        
    
    def mse(self,y_act,y_pred):
        sumation = 0 
        for i, j in zip(y_act,y_pred):
            sumation += (i-j)**2
        print(sumation/len(y_act))


