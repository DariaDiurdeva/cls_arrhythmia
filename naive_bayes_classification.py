from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from data_preparation import prepare_data


def classify(X, Y, test_size=0.30):
    # разделим датасет на тренировочные и тестовые данные
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=89)
    # создадим экземпляр класса классификатора GaussianNB
    nb_model = GaussianNB()
    # обучим классификатор с помощью функции fit на тренеровочных данных
    nb_model.fit(x_train, y_train)
    # применим классификатор к тестовым данных
    prediction = nb_model.predict(x_test)
    return prediction, y_test


# обратимся к функции
X, Y = prepare_data()
predict, real_values = classify(X, Y)
# Выводим метрики оценки качества классификатора
print(accuracy_score(predict, real_values))
print(classification_report(predict, real_values))
