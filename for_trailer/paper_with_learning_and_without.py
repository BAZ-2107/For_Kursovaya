# -*- coding: utf-8 -*-


from scipy.integrate import solve_ivp
import numpy as np
from scipy.linalg import solve_continuous_are
import pygame
import sympy as sp
import matplotlib.pyplot as plt


GRAY = (125, 125, 125)
BLACK=(0, 0, 0)
RED = (255, 0, 0)


class ExplorationNoise:
    def __init__(self, m, n_frequencies=8, amplitude=.005, freq_range=5.):
        self.m = m
        self.amplitude = amplitude
        self.omega = np.random.uniform(-freq_range, freq_range, (m, n_frequencies))
    
    def __call__(self, t):
        return self.amplitude * np.sum(np.sin(self.omega * t), axis=1)


class Trailer:
    x, y, theta, gamma = None, None, None, None
    xmin, xmax, ymin, ymax = None, None, None, None
    x1, y1, x2, y2, x3, y3, x4, y4 = None, None, None, None, None, None, None, None

    def __init__(self, L=6., D=3, h=2.6, d=1.):
        self.L, self.D, self.h, self.d = L, D, h, d

    def setParams(self, *args):
        if args:
            self.x = args[0]
        if len(args) > 1:
            self.y = args[1]
        if len(args) > 2:
            self.theta = args[2]
        if len(args) > 3:
            self.gamma = args[3]

    def getParams(self):
        return self.x, self.y, self.theta, self.gamma

    def getCoords(self):
        return -self.y, self.x

    def setField(self, xmin, xmax, ymin, ymax):
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax

    def getField(self):
        return np.array([[self.xmin, self.ymin], [self.xmin, self.ymax],
                         [self.xmax, self.ymax], [self.xmax, self.ymin]])

    def setBorders(self, x1, y1, x2, y2, x3, y3, x4, y4):
        self.x1, self.y1, self.x2, self.y2, self.x3, self.y3, self.x4, self.y4 = x1, y1, x2, y2, x3, y3, x4, y4

    def getBorders(self):
        return np.array([[self.x1, self.y1], [self.x2, self.y2], [self.x3, self.y3], [self.x4, self.y4]])

    def isintersect(self, a1, b1, a2, b2, c1, d1, c2, d2):
        k1, k2, k3, k4 = a1 - a2, c2 - c1, b1 - b2, d2 - d1
        DD = k1 * k4 - k3 * k2
        if DD != 0.:
            k5, k6 = c2 - a2, d2 - b2
            return (0 <= ((k1 * k6 - k3 * k5) / DD) <= 1) and (0 <= ((k5 * k4 - k2 * k6) / DD) <= 1)
        return False

    def check_stop(self, borders=False):
        arr = self.getPoints() + np.array([-self.y, self.x])
        arr_2 = self.getBorders()
        arr_3 = self.getField()
        intervals = [[arr[0], arr[1]], [arr[0], arr[2]], [arr[1], arr[3]], [arr[2], arr[3]],
                     [arr[4], arr[5]], [arr[4], arr[6]], [arr[5], arr[7]], [arr[6], arr[7]]]
        if borders:
            intervals_2 = [[arr_2[0], arr_2[1]], [arr_2[2], arr_2[3]], [arr_3[0], arr_3[1]],
                       [arr_3[1], arr_3[2]], [arr_3[2], arr_3[3]], [arr_3[0], arr_3[3]]]
        else:
            intervals_2 = [[arr_3[0], arr_3[1]], [arr_3[1], arr_3[2]], [arr_3[2], arr_3[3]], [arr_3[0], arr_3[3]]]
        
        for x, y in intervals:
            for z, w in intervals_2:
                if self.isintersect(*x, *y, *z, *w):
                    return True
        return False

    def getPoints(self):
        x, y, L, h2, d, D = -self.y, self.x, self.L, self.h / 2, self.d, self.D
        truck_or, trailer_or = self.theta, self.theta + self.gamma
        
        truck = np.array([[-h2, D + d], [h2, D + d], [-h2, -d], [h2, -d]])
        trailer = np.array([[-h2, d], [h2, d], [-h2, -L - d], [h2, -L - d]])
        truck_orientation = np.vstack([truck @ np.array([np.cos(truck_or), -np.sin(truck_or)]), truck @ np.array([np.sin(truck_or), np.cos(truck_or)])]).T
        trailer_orientation = np.vstack([trailer @ np.array([np.cos(trailer_or), -np.sin(trailer_or)]), trailer @ np.array([np.sin(trailer_or), np.cos(trailer_or)])]).T
        return np.vstack([truck_orientation, trailer_orientation])


e = ExplorationNoise(1)
L, D = 6., 3.
L_real, D_real = 11., 3.
A = np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., -1. / L]])
B = np.array([[0.], [1 / D], [-1 / D]])
Q = np.eye(3)
R = np.array([[1000.]])
R_real = np.array([[100.]])


P = solve_continuous_are(A, B, Q, R)
P_2 = solve_continuous_are(-A, -B, Q, R)
K_f = np.dot(np.linalg.inv(R), np.dot(np.transpose(B), P))
K_b = np.dot(np.linalg.inv(R), np.dot(np.transpose(-B), P_2))
K_bb = K_b.copy()
#print(K_f)
#print(K_b)


def coords_for_u(currr, targg): # изменение системы координат для управления
    yy = (targg[0] - currr[0]) * np.sin(targg[2]) + (currr[1] - targg[1]) * np.cos(targ[2])
    ttheta = currr[2] - targg[2]
    ggamma = currr[-1]
    return np.array([yy, ttheta, ggamma])


def vdp(t, vec, D, L, x_targ): # интегрирование системы(система для движения вперед)
    x, y, theta, gamma = vec
    u = (-K_f @ coords_for_u(vec, x_targ))[0] # управление --- скаляр
    return [np.cos(theta), np.sin(theta), u / D, -1 / L * np.sin(gamma) - u / D]


def vdp_back(t, vec, D, L, x_targ): # интегрирование необходимых переменных(система для движения назад)
    x, y, theta, gamma = vec[:4]
    X = vec[1:4]
    e_t = e(t)
    u = (-K_b @ coords_for_u(vec[:4], x_targ) + e_t)[0]
    arr = np.array([-np.cos(theta), -np.sin(theta), -u / D, 1 / L * np.sin(gamma) + u / D])
    sigma = np.array([theta - np.sin(theta), np.sin(gamma) - gamma])
    r_k = X.T @ (Q + K_b.T @ R_real @ K_b) @ X
    return np.hstack([arr, -2 * np.kron(sigma, X), -2 * np.kron(R_real @ e_t, X), np.array([-r_k])])


def vdp_back_2(t, vec, D, L, x_targ): # интегрирование необходимых переменных(система для движения назад)
    x, y, theta, gamma = vec
    u = (-K_b @ coords_for_u(vec, x_targ))[0]
    arr = np.array([-np.cos(theta), -np.sin(theta), -u / D, 1 / L * np.sin(gamma) + u / D])
    return arr


def get_arr(trailer, size_scr, scale=10): # получение массива точек для отрисовки тягача и прицепа
    x, y = trailer.getCoords()
    arr = t.getPoints() * scale
    arr[:, 1] = -arr[:, 1]
    X, Y = size_scr[0] // 2, size_scr[1] // 2
    arr[:, 0] += X + x * scale
    arr[:, 1] += Y - y * scale
    a1, a2, a3, a4, a5, a6, a7, a8 = arr
    return np.array([a1, a2, a4, a3, a5, a6, a8, a7])

def get_Borders(trailer, size_scr, scale=10): # получение координат бордюров
    arr = trailer.getBorders() * scale
    arr[:, 1] = -arr[:, 1]
    X, Y = size_scr[0] // 2, size_scr[1] // 2
    arr[:, 0] += X
    arr[:, 1] += Y
    return arr

def get_Field(trailer, size_scr, scale=10): # получение координат прямоугольной области
    arr = trailer.getField() * scale
    arr[:, 1] = -arr[:, 1]
    X, Y = size_scr[0] // 2, size_scr[1] // 2
    arr[:, 0] += X
    arr[:, 1] += Y
    return arr

def draw_car(pg, sc, t, sz_sc, scale=10): # отрисовка тягача и прицепа
    arr = get_arr(t, sz_sc, scale)
    pg.draw.polygon(sc, GRAY, list(arr[:4]))
    pg.draw.polygon(sc, BLACK, arr[4:])


def draw_field(pg, sc, t, sz_sc, scale=10): # Отрисовка прямоугольной арены
    arr = get_Borders(t, sz_sc, scale)
    arr_2 = get_Field(t, sz_sc, scale)
    pg.draw.line(sc, RED, arr[0], arr[1], 2)
    pg.draw.line(sc, RED, arr[2], arr[3], 2)
    pg.draw.polygon(sc, GRAY, arr_2, 2)


def get_v_for_dxx(vec): # получение \overline{x} из x
    arr, n = [], len(vec)
    for i in range(n):
        for j in range(i, n):
            arr.append(vec[i] * vec[j])
    return np.array(arr)


def get_Pk(vector, n): # получение матрицы Pk из \hat(P_k)
    Pkk = np.zeros((n, n))
    k = 0
    for i in range(n):
        for j in range(i, n):
            pij = vector[k]
            if i != j:
                pij /= 2
            Pkk[i][j] = pij
            if i != j:
                Pkk[j][i] = pij
            k += 1
    return Pkk


def repeated_code(pygame, screen, t, size, fps):
    screen.fill((255, 255, 255))
    draw_car(pygame, screen, t, size, 10)
    draw_field(pygame, screen, t, size, 10)
    clock.tick(fps)
    pygame.display.flip()
    return 


#if __name__ == "__main__"
pygame.init()
is_borders = True
size = 1200, 720
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Трейлер")
clock = pygame.time.Clock()
exit = False

pygame.display.flip()
Delta_A = np.array([[1., 0.], [0., 0.], [0., 1 / L]])

t = Trailer(L_real, D_real)
t.setField(-15., 15., -15., 35.) # задается прямоугольная область
t.setBorders(-2., -15., -2., 4., 2., -15., 2., 4.) # задаются бордюры

curr = np.array([0., -5., .0, 0.]) # Текущее состояние системы

x2 = np.array([20., 0., 0., 0.])
x1 = np.array([20., 2., .3, 0.])


goal = np.zeros(4)
targ = x1

cnt = 0
time = 40
ccnt = 161
fps  = 100
for_plot_2 = list()

while not exit:
    cnt += 1
    if cnt > 16:
        break
    sol = solve_ivp(vdp, [0., time], curr, t_eval=np.linspace(0., time, ccnt), args=(D_real, L_real, targ)).y.T
    for item in sol:
        t.setParams(*item)
        if t.check_stop(is_borders):
            break
        curr = item
        for_plot_2.append(curr)
        repeated_code(pygame, screen, t, size, fps)
    soll = solve_ivp(vdp_back_2, [0., time], curr, t_eval=np.linspace(0., time, ccnt), method='DOP853', rtol=1e-12, atol=1e-14, args=(D_real, L_real, goal)).y.T
    for index, item in enumerate(soll):
        t.setParams(*item)
        if t.check_stop(is_borders) or abs(item[3]) > .9:
            break
        curr = item
        for_plot_2.append(curr)
        repeated_code(pygame, screen, t, size, fps)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit = True
            break
    curr_y = curr[1]
    if curr_y >= .3:
        targ = x1
    elif curr_y <= .3 and abs(curr_y) >= .05:
        targ = x2
    else:
        if curr[0] < 4.:
            exit = True
            break
exit = False
while not exit:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                exit = True
                break
print(cnt, "ИТЕРАЦИЙ! --- Без обучения")


curr = np.array([0., -5., .0, 0.]) # Текущее состояние системы

goal = np.zeros(4)
targ = x1

for_plot = list()

P_k_back = None
exit = False
flag_for_P = False
cnt = 0
time = 40
ccnt = 161
fps  = 100
while not exit:
    cnt += 1
    if cnt > 16 :
        break
    sol = solve_ivp(vdp, [0., time], curr, t_eval=np.linspace(0., time, ccnt), args=(D_real, L_real, targ)).y.T
    for item in sol:
        t.setParams(*item)
        if t.check_stop(is_borders):
            break
        curr = item
        for_plot.append(curr)
        repeated_code(pygame, screen, t, size, fps)
    soll = solve_ivp(vdp_back, [0., time], np.hstack([curr, np.zeros(10)]), t_eval=np.linspace(0., time, ccnt), method='DOP853', rtol=1e-12, atol=1e-14, args=(D_real, L_real, goal)).y
    sol_2 = soll[:4].T
    sol_3 = soll[4:].T
    Theta = list()
    Xi = list()
    rank = 0
    flag_for_theta = False
    x_for_kron = sol_2[0][1:]
    for index, item in enumerate(sol_2):
        t.setParams(*item)
        if t.check_stop(is_borders) or abs(item[3]) > .9:
            break
        if flag_for_theta:
            Theta.append(np.hstack([get_v_for_dxx(item[1:]) - get_v_for_dxx(x_for_kron),
                                 sol_3[index][:-1] - sol_3[index - 1][:-1]]))
            Xi.append(sol_3[index][-1] - sol_3[index - 1][-1])
            x_for_kron = item[1:]
        if index == 0:
            flag_for_theta = True
        curr = item
        for_plot.append(item)
        repeated_code(pygame, screen, t, size, fps)
    np_Theta = np.array(Theta.copy())
    np_Xi = np.array(Xi.copy())
    rank = np.linalg.matrix_rank(np_Theta)
    if rank == 15:
        vect = np.linalg.pinv(np_Theta, rcond=None) @ np_Xi
        Pk = get_Pk(vect, 3)
        if not flag_for_P:
            flag_for_P = True
        else:
            err = np.linalg.norm(Pk - P_k_back)
            print(err)
            if err < .001:
                exit = True
            #print("Error of Pk: ", np.linalg.norm(Pk - P_k_back))
        #print(Pk)
        P_k_back = Pk.copy()
        K_b = np.array([vect[-3:]]).copy()
        #print(K_b)
    Theta.clear()
    Xi.clear()
    #print(K_b)
    curr_y = curr[1]
    targ = x1 if curr_y >= .3 else x2

exit = False
while not exit:
    cnt += 1
    if cnt > 16:
        break
    sol = solve_ivp(vdp, [0., time], curr, t_eval=np.linspace(0., time, ccnt), args=(D_real, L_real, goal)).y.T
    for item in sol:
        t.setParams(*item)
        if t.check_stop(is_borders):
            break
        curr = item
        for_plot.append(curr)
        repeated_code(pygame, screen, t, size, fps)
    soll = solve_ivp(vdp_back_2, [0., time], curr, t_eval=np.linspace(0., time, ccnt), method='DOP853', rtol=1e-12, atol=1e-14, args=(D_real, L_real, goal)).y.T
    for item in soll:
        t.setParams(*item)
        if t.check_stop(is_borders) or abs(item[3]) > .9:
            break
        curr = item
        for_plot.append(curr)
        repeated_code(pygame, screen, t, size, fps)
    curr_y = curr[1]
    if curr_y >= .3:
        targ = x1
    elif curr_y <= .3 and abs(curr_y) >= .05:
        targ = x2
    else:
        if curr[0] < 4.:
            exit = True
            break
exit = False
while not exit:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit = True
            break
print(cnt, "ИТЕРАЦИЙ! --- С обучением")
print("K_b_effective: ", K_b)

pygame.quit()

step = time / (ccnt - 1)
countt = len(for_plot)
t_eval = np.linspace(0., step * countt, countt)
countt_2 = len(for_plot_2)
t_eval_2 = np.linspace(0., step * countt_2, countt_2)
np_arr = np.array(for_plot).T
np_arr_2 = np.array(for_plot_2).T
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 6))

for i, label, title in zip([0, 1, 2, 3], ["x(t)", "y(t)", "theta(t)", "gamma(t)"], ["Продольное смещение", "Поперечное смещение", "Ориентация тягача", "Угол между тягачом и прицепом"]):
    for lab, arr, ttt in zip(["No learning", "Learning with ADP"], [np_arr_2, np_arr], [t_eval_2, t_eval]):
        axes[i].plot(ttt, arr[i], label=lab)
    axes[i].set_title(title)
    axes[i].set_xlabel('t')
    axes[i].set_ylabel(label)
    axes[i].legend()
    axes[i].grid(True)
plt.tight_layout()
plt.show()