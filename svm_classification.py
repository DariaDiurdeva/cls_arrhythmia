from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from data_preparation import prepare_data

X, Y = prepare_data()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30,random_state=42)
svc_model = SVC(C=0.1,kernel = 'linear', random_state = 0)
svc_model.fit(x_train, y_train)
prediction = svc_model.predict(x_test)

print(y_test)
print(prediction)

print(accuracy_score(prediction, y_test))
print(classification_report(prediction, y_test))