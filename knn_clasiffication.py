from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from data_preparation import prepare_data

NUMBER_OF_NEIGHBORS = 20
X, Y = prepare_data()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=0)
knn_model = KNeighborsClassifier(NUMBER_OF_NEIGHBORS, weights='distance')
knn_model.fit(x_train, y_train)
prediction = knn_model.predict(x_test)

print(y_test)
print(prediction)