from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from data_preparation import prepare_data


X, Y = prepare_data()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30,random_state=0)
naiveBayes_model = GaussianNB()
naiveBayes_model.fit(x_train, y_train)
prediction = naiveBayes_model.predict(x_test)

print(y_test)
print(prediction)

print(accuracy_score(prediction, y_test))
print(classification_report(prediction, y_test))

