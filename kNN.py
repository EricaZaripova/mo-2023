# 1. Датасет – ирис (либо скачать с сайта kaggle, либо вытащить с библиотек sklearn (именно само множество)).
# 2. Нормализовать данные (самостоятельно). Разделить выборку на две части: обучающую и тестовую.
# При помощи этого разделения подобрать оптимальное количество соседей k (самостоятельно).
# 3. Вывести проекции на оси (6 разных картинок, минимум) – причем как до нормализации,
# так и после (по итогу, минимум 12 картинок).
# 4. Задавать новый объект (с клавиатуры, файла, нажатием на экран или где-то в коде)
# и определять его класс – нахождением k его ближайших соседей.

import sys
import matplotlib.pyplot as plt
import numpy as np
import random
from itertools import combinations
from sklearn.datasets import load_iris


# вывод графиков
def plot_iris(dataset):
    X = dataset['data']
    y = dataset['target']

    fig, axs = plt.subplots(2,3)
    comb = list(combinations((0,1,2,3), 2))
    col_names = ["sepal length", "sepal width", "petal length", "petal width"]

    for i in range(2):
        for j in range(3):
            axs[i][j].scatter(X[:,comb[i*3+j][0]], X[:,comb[i*3+j][1]],c = y)
            axs[i][j].set_title(f"projection on {col_names[comb[i*3+j][0]]} and {col_names[comb[i*3+j][1]]}")
            axs[i][j].set_xlabel(col_names[comb[i*3+j][0]])
            axs[i][j].set_ylabel(col_names[comb[i*3+j][1]])

    plt.show()


# минимальные значения параметров для нормальзации
def min_dataset_param(data, index):
    min_param = sys.maxsize
    for i in range(len(data)):
        if data[i][index] < min_param:
                min_param = data[i][index]

    return min_param


# максимальные значения параметров для нормальзации
def max_dataset_param(data, index):
    max_param = -float('inf')
    for i in range(len(data)):
        if data[i][index] > max_param:
                max_param = data[i][index]

    return max_param


# расстояние между точками
def dist(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


# определение принадлежности к классу для элемента
def knn_class(item, dataset, k):
    item_data = item[0]
    # расстояния до соседей
    distances = [(data, np.linalg.norm(np.array(item_data) - np.array(data[0]))) for data in dataset]
    sorted_distances = sorted(distances, key=lambda x: x[1])
    k_nearest_data = [pair[0] for pair in sorted_distances[:k]]

    # считаем какого класса среди к соседей больше
    class_frequencies = {}
    for data, class_label in k_nearest_data:
        if class_label in class_frequencies:
            class_frequencies[class_label] += 1
        else:
            class_frequencies[class_label] = 1
    most_common_class = max(class_frequencies, key=class_frequencies.get)

    return most_common_class


# поиск оптимального числа соседей
def optimal_k(train_dataset, test_dataset):
    n = len(train_dataset) + len(test_dataset)
    optimal = 1
    best_accuracy = 0

    # для каждого количества соседей проверяем точность определения
    for k in range(1, int(np.sqrt(n))):
        matches = 0
        for item in test_dataset:
            item_class = knn_class(item, train_dataset, k)
            if item_class == item[1]:
                matches += 1
        accuracy = matches / len(test_dataset)
        if accuracy > best_accuracy:
            optimal = k
            best_accuracy = accuracy

    return optimal, best_accuracy


# нормализация данных iris
def normalize_iris(dataset):
    normalized_dataset = dataset
    data = normalized_dataset.data
    for j in range(4):
        min_param = min_dataset_param(data, j)
        max_param = max_dataset_param(data, j)
        for i in range(len(data)):
            data[i][j] = (data[i][j] - min_param) / (max_param - min_param)

    return normalized_dataset


# нормализация значений элемента
def normalize_item(item, dataset):
    for i in range(4):
        min_param = min_dataset_param(dataset.data, i)
        max_param = max_dataset_param(dataset.data, i)
        item[i] = min((item[i] - min_param) / (max_param - min_param), 1)
    return item


if __name__ == '__main__':
    # графики с данными из базы
    iris = load_iris()
    # plot_iris(iris)
    # # графики с нормализованными данными
    norm_iris = normalize_iris(iris)
    # plot_iris(iris)

    # создание массива с элементами
    items = []
    for i in range(len(norm_iris.data)):
        items.append([norm_iris.data[i], norm_iris.target[i]])

    # создание массивов с тренировочными и тестовыми данными
    test_len = 20
    train_dataset = items[test_len:]
    test_dataset = items[:test_len]

    k, accuracy = optimal_k(train_dataset, test_dataset)
    print(f'Оптимальное число соседей: {k}, Точность: {accuracy*100}%')

    item = [4.9, 3, 1.4, 0.2] # setosa
    # item = [5, 2.3, 3.3, 1] # versicolor
    # item = [6.2, 3.4, 5.4, 2.3] # virginica

    iris = load_iris()
    item = normalize_item(item, iris)
    cl = knn_class(item, train_dataset, k)
    print(f'Добавленный элемент принадлежит к классу {iris.target_names[cl]}')