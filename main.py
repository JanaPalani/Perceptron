import numpy as np
from perceptron import Perceptron
# [TESTING]


# CREATED DATA 
# data_x =[]
# data_y =[]
# for i in range(200):
#     j = random.randint(10,20)
#     n = random.randint(10,20)
#     l = random.randint(10,20)
#     k = 0.5*j + 0.25*n - 0.6*l -0.324
#     data_x.append([j,n,l])
#     data_y.append(k)
# data_y = np.array(data_y)
# data_x = np.array(data_x)
# data_y = np.reshape(data_y,(len(data_y),1))


# DIABETES DATA SET 
from sklearn.datasets import load_diabetes
data = load_diabetes()
data_x = data.data
data_y = data.target
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
data_x=sc.fit_transform(data_x)
data_y = np.reshape(data_y,(len(data_y),1))
data_y=sc.fit_transform(data_y)
        

# MAIN 
alpha = float(input('enter the learning rate:'))
neuron = Perceptron(alpha)
neuron.train(data_x,data_y,10000)
predicted = neuron.predict(data_x)
predicted = sc.inverse_transform(predicted)
data_y = sc.inverse_transform(data_y)
c = np.concatenate([data_y,predicted],axis = 1)
print(c)
neuron.mse(data_y,predicted)



