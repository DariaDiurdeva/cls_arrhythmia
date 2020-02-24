from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from data_preparation import prepare_data

NUMBER_OF_NEIGHBORS = 10


def classify(x, y, neighbors=NUMBER_OF_NEIGHBORS, weights='distance', test_size=0.30):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=23)
    knn_model = KNeighborsClassifier(neighbors, weights=weights)
    knn_model.fit(x_train, y_train)
    prediction = knn_model.predict(x_test)
    return prediction, y_test


X, Y = prepare_data()
predict, real_values = classify(X, Y)

print(accuracy_score(predict, real_values))
print(classification_report(predict, real_values))
