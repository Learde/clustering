import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

POINT_N = 300
DIM_N = 2
CLUST_N = 3

data, target, Center = datasets.make_blobs(n_samples=POINT_N, centers=CLUST_N, n_features=DIM_N, center_box=(-10, 10),
                                           random_state=0, return_centers=True)

slope = (Center[1][1] - Center[0][1]) / (Center[1][0] - Center[0][0])
C0 = [(Center[0][0] + Center[1][0]) / 2, (Center[0][1] + Center[1][1]) / 2]

print('\nMulticluster')

A = 1.0


def sigma(x):
    #return 1.0 / (1.0 + np.exp(-A * x))
    # Преобразуем вектор в массив, чтобы избежать ошибок типа "integer division or modulo by zero"
    z = np.array(x)
    # Вычисляем экспоненты каждого элемента вектора
    exp_z = np.exp(z)
    # Вычисляем сумму экспонент всех элементов вектора
    sum_exp_z = np.sum(exp_z)
    # Вычисляем вероятности для каждого элемента вектора
    softmax_z = exp_z / sum_exp_z
    return softmax_z


def delta(x, c):
    return np.sum(np.linalg.norm(x - c, axis=1))

"""
Здесь x - это массив координат точек, а c - массив координат центров кластеров. np.linalg.norm вычисляет евклидово расстояние между точками и центрами кластеров
"""

np.random.seed(5)


class NNET4:
    def __init__(self):
        self.input_nodes = DIM_N
        self.output_nodes = CLUST_N
        self.weights_input_to_output = np.random.rand(self.output_nodes, self.input_nodes) # матрица весов (число выходных * число входных)
        self.output_bias = np.zeros(self.output_nodes, dtype=float) # вектор смещения

    def run(self, features):
        return sigma(np.dot(self.weights_input_to_output, features) + self.output_bias) # скалярное произведение матрицы весов на вектор признаков (выдает массив с предсказаниями нейронов)


network = NNET4()

trg = [[int(target[n] == i) for i in range(CLUST_N)] for n in range(POINT_N)] # в какой кластер пойдет точка

u = np.empty(CLUST_N, dtype=float)
v = np.empty(CLUST_N, dtype=float)
p = np.empty(CLUST_N, dtype=float)

H = 0.1  # шаг
eps = 0.01
for iter in range(300):
    sw = np.zeros([CLUST_N, DIM_N], dtype=float)
    sb = np.zeros(CLUST_N, dtype=float)
    for n in range(POINT_N):
        u = np.dot(network.weights_input_to_output, data[n]) + network.output_bias
        v = trg[n] - sigma(u)
        p = v * delta(data[n].reshape(1, -1), Center) * sigma(u) * (1.0 - sigma(u))
        sb += p
        for k in range(CLUST_N):
            sw[k] += p[k] * data[n]
    network.output_bias += H * sb
    network.weights_input_to_output += H * sw
    if np.linalg.norm(sw) < eps and np.linalg.norm(sb) < eps:
        break

print('eps=', eps, '     iter=', iter)

pred = []
for n in range(POINT_N):
    res = network.run(data[n])
    r0 = 0.0
    i0 = 0
    for i in range(CLUST_N):
        if res[i] > r0:
            i0 = i
            r0 = res[i]
    pred.append(i0)

print(1.0 * sum(pred == target) / POINT_N)
print(network.weights_input_to_output, network.output_bias)

Color = ['blue', 'green', 'cyan', 'black']

# Отобразить точки, разделенные по кластерам
plt.figure(figsize=(8, 8))
for i in range(POINT_N):
    plt.scatter(data[i][0], data[i][1], c=Color[pred[i]], marker='o')

# Отобразить центры кластеров
plt.scatter(Center[:, 0], Center[:, 1], c='red', marker="*", s=100)

plt.show()
