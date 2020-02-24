from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from data_preparation import prepare_data


def classify(x, y, test_size=0.30):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=23)
    svm_model = SVC(C=0.1, kernel='linear', gamma=0.7)
    svm_model.fit(x_train, y_train)
    prediction = svm_model.predict(x_test)
    return prediction, y_test


X, Y = prepare_data()
predict, real_values = classify(X, Y)

print(accuracy_score(predict, real_values))
print(classification_report(predict, real_values))



