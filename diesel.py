 # -*- coding: utf-8 -*-


import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp, quad_vec
from scipy.linalg import solve_continuous_are


class ExplorationNoise: # Класс, генерирующий шум
    def __init__(self, m, n_frequencies=100, amplitude=100, freq_range=500):
        self.m = m
        self.amplitude = amplitude
        self.omega = np.random.uniform(-freq_range, freq_range, (m, n_frequencies))
    
    def __call__(self, t):
        return self.amplitude * np.sum(np.sin(self.omega * t), axis=1)


def print_matrix(M, acc=4): # печать матрицы в нормальном формате(с округлением)
    m, n = M.shape
    for i in range(m):
        for j in range(n):
            print(str(f"{M[i][j]:.{acc}f}"), end=" ")
        print()


def exact_rank_sympy(matrix): # функция определения ранга матрицы
    M_sym = sp.Matrix(matrix.tolist())
    rank = M_sym.rank()
    return rank


def get_Pk(vector, n): # получение матрицы Pk из \hat(P_k)
    Pk = np.zeros((n, n))
    k = 0
    for i in range(n):
        for j in range(i, n):
            pij = vector[k]
            if i != j:
                pij /= 2
            Pk[i][j] = pij
            if i != j:
                Pk[j][i] = pij
            k += 1
    return Pk


A = np.array([[-.4125, -.0248, .0741, .0089, 0., 0.],
              [101.5873, -7.2651, 2.7608, 2.8068, 0., 0.],
              [.0704, .0085, -.0741, -.0089, 0., .02],
              [.0878, .2672, 0., -.3674, .0044, .3962],
              [-1.8414, .099, 0., 0., -.0343, -.0330],
              [0., 0., 0., -359., 187.5364, -87.0316]])
n = A.shape[0]

B = np.array([[-.0042, .0064], [-1.036, 1.5849], [.0042, 0.],
              [.1261, 0.], [0., -.0168], [0., 0.]])
m = B.shape[-1]

Q = np.diag([1., 1., .1, .1, .1, .1])

R = np.eye(2)

P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

print("Матрица P*")
print_matrix(P)
print()
print("Матрица K*")
print_matrix(K)
print()

K0 = np.zeros((m, n))
eigenvalues = np.linalg.eigvals(A - B @ K0)
if not np.all(np.real(eigenvalues) < 0): # Проверка, что K0 --- стабилизирующая
    raise Exception("Матрица K0 не является стабилизирующей для пары матриц (A, B)!")


e = ExplorationNoise(m)


def func(t, y): # функция интегрирования x, x\otimes x, x \otimes u
    x = y[:n]
    u = -K0 @ x + e(t)
    for_Ixx = np.kron(x, x)
    for_Ixu = np.kron(x, u)
    return np.hstack([A @ x + B @ u, for_Ixx, for_Ixu])


def get_v_for_dxx(vec): # получение \overline{x} из x
    arr, n = [], len(vec)
    for i in range(n):
        for j in range(i, n):
            arr.append(vec[i] * vec[j])
    return np.array(arr)

delta = 0.01
count = 50

dxx, Ixx, Ixu = list(), list(), list()
x0 = np.hstack([np.random.randn(A.shape[-1]), np.zeros(n * (n + m))])
curr_t0 = 0.
pred_over_x = get_v_for_dxx(x0[:n])
rang = 0

for i in range(1000): # сбор данных для формирования матриц Ixx, Ixu, \delta xx
    if i and (i % 40) == 0:
        rang = exact_rank_sympy(np.hstack([np.array(Ixx), np.array(Ixu)]))
        if rang == n * (n + 1) // 2 + m * n:
            break
    if i > 100:
        raise Exception("Цикл улетел!")
    sol = solve_ivp(func, (curr_t0, curr_t0 + delta), x0, t_eval=np.array([curr_t0, curr_t0 + delta]), method='DOP853', rtol=1e-12, atol=1e-14).y

    curr_over_x = get_v_for_dxx(sol[:n, 1])
    dxx.append(curr_over_x - pred_over_x)
    pred_over_x = curr_over_x

    Ixx.append(sol[n:n + n * n, 1] - sol[n:n + n * n, 0])
    Ixu.append(sol[-m * n:, 1] - sol[-m * n:, 0])

    curr_t0 += delta
    x0 = sol[:, 1]


counter = 0
presicion = .03
Kk = K0
Kkk = K0
Ixx, Ixu, dxx = np.array(Ixx), np.array(Ixu), np.array(dxx)
Pk, Pkk = np.zeros_like(Q), np.zeros_like(Q)

while True: # итерационный процесс нахождения Pk и Kk+1
    counter += 1
    Theta = np.hstack([dxx, -2. * (Ixx @ np.kron(np.eye(n), (Kk.T @ R)) + (Ixu @ np.kron(np.eye(n), R)))])
    Qk = Q + (Kk.T @ R @ Kk)
    Xi = -Ixx @ Qk.flatten(order='F')
    vec = np.linalg.pinv(Theta, rcond=None) @ Xi
    Kkk = np.transpose(np.array(vec[-m * n:]).reshape((n, m)))    
    if counter == 1:
        Pk = get_Pk(vec, n)
        Kk = Kkk
        continue
    Pkk = get_Pk(vec, n)
    if np.linalg.norm(Pk - Pkk) <= presicion:
        break
    if counter > 1000:
        raise Exception("Сходимость ушла в закат!")
    Pk = Pkk
    Kk = Kkk

print("k =", counter)
print("Матрица Kk+1:")
print_matrix(Kkk)
print()
print("Матрица Pk:")
print_matrix(Pkk)