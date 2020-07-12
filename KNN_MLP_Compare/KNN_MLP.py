from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from KNN_MLP_Compare.Test_data import get_data

X,Y = get_data()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size
= 0.2, random_state = 0)

"""KNN"""
# Instantiate learning model (k = 3) and fit model
y_pred = KNeighborsClassifier(n_neighbors=3).fit(X_train,
                                                 y_train).predict(X_test)
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of KNN is equal {} %.'.format(round(accuracy, 2)))

"""MLP"""
predictions = MLPClassifier(hidden_layer_sizes=(13,13,13),
                            max_iter=10).fit(X_train,y_train).predict(X_test)
accuracy = accuracy_score(y_test, predictions)*100
print('Accuracy of MLP is equal {} %.'.format(round(accuracy, 2)))

"""
Result: the neural network is less accurate than the kNN.
"""