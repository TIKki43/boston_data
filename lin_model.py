# Пример регрессии
import matplotlib.pyplot as plt  # для графика
from pandas import read_csv  # для работы с датасетом
from keras.models import Sequential  # для создания стека со слоями
from keras.layers import Dense  # импортируем полносвязные слои
from keras.wrappers.scikit_learn import KerasRegressor  # для построения регрессии
from sklearn.model_selection import cross_val_score  # для оцен. результата с помощью cross_val
from sklearn.model_selection import KFold  # для оценки
# загружаем датасет в переменную
dataframe = read_csv("housing.csv", delim_whitespace=True, header=None)
# можете посмотреть результат
print(dataframe)
# берем из всей таблицы только значения
dataset = dataframe.values
# можете посмотреть, что получилось
print(dataset)
# делим все данные на X и Y, т.е. X - данные, основываясь на которых нн должна сделать предикт, а Y - сам верный предикт
X = dataset[:, 0:13]
Y = dataset[:, 13]


# создаем функцию с моделью
def baseline_model():
    # строим модель
    model = Sequential()
    # добавляем входной слой
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    # добавляем выходной слой
    model.add(Dense(1, kernel_initializer='normal'))
    # компилируем модель с функцией ошибок MSE и оптимизатором adam
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# обучение модели
history = baseline_model().fit(X, Y, epochs=500, verbose=0)

# построение графика
plt.plot(history.history['loss'])
plt.grid(True)
plt.show()  