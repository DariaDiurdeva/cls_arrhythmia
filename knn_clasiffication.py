
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from data_preparation import prepare_data

NUMBER_OF_NEIGHBORS = 13
X, Y = prepare_data()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
knn_model = KNeighborsClassifier(NUMBER_OF_NEIGHBORS, weights='distance')
knn_model.fit(x_train, y_train)
prediction = knn_model.predict(x_test)

print(y_test)
print(prediction)

print(accuracy_score(prediction, y_test))
print(classification_report(prediction, y_test))

