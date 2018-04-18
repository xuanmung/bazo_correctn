import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from matplotlib import pyplot as plt
import seaborn as sns

model = Sequential([Dense(output_dim=4, input_dim=1),
                    Activation("tanh"),
                    Dense(output_dim=4, input_dim=4),
                    Activation("linear"),
                    Dense(output_dim=1, input_dim=4)])

model.compile(loss='mse', optimizer='nadam')

#X = np.array([[i] for i in range(100)])
#y = np.array([3*x[0]+(x[0]**2)+1 for x in X]) + np.random.normal(0, 1, 100)

dataset = np.loadtxt("/media/sclee/A8D41CC7D41C99A0/XUANMUNG/STUDY/HOC_NEURAL_NET/data_v_dh.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0]
y = dataset[:,1]

model.fit(X, y, nb_epoch=500)
y_predicted = model.predict(X)
plt.scatter(X.reshape(-1, 1), y)
plt.plot(X.reshape(-1, 1), y_predicted)
print(y_predicted[1])
print(y_predicted[150])
plt.show()