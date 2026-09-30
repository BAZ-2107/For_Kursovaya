# -*- coding: utf-8 -*-


from scipy.integrate import solve_ivp
import numpy as np
from scipy.linalg import solve_continuous_are
import pygame
import sympy as sp


GRAY = (125, 125, 125)
BLACK=(0, 0, 0)


class ExplorationNoise:
    def __init__(self, m, n_frequencies=8, amplitude=.005, freq_range=5.):
        self.m = m
        self.amplitude = amplitude
        self.omega = np.random.uniform(-freq_range, freq_range, (m, n_frequencies))
    
    def __call__(self, t):
        return self.amplitude * np.sum(np.sin(self.omega * t), axis=1)


e = ExplorationNoise(1)


class Trailer:
    border = 1.5
    x, y, theta, gamma = None, None, None, None
    xmin, xmax, ymin, ymax = None, None, None, None
    x1, y1, x2, y2, x3, y3, x4, y4 = None, None, None, None, None, None, None, None

    def __init__(self, LL=6., D=3, h=2.6, d=1.):
        self.L, self.D, self.h, self.d = LL, D, h, d

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

    def check_stop(self):
        arr = self.getPoints() + np.array([-self.y, self.x])
        arr_2 = self.getBorders()
        arr_3 = self.getField()
        intervals = [[arr[0], arr[1]], [arr[0], arr[2]], [arr[1], arr[3]], [arr[2], arr[3]],
                     [arr[4], arr[5]], [arr[4], arr[6]], [arr[5], arr[7]], [arr[6], arr[7]]]
        intervals_2 = [[arr_2[0], arr_2[1]], [arr_2[2], arr_2[3]], [arr_3[0], arr_3[1]],
                       [arr_3[1], arr_3[2]], [arr_3[2], arr_3[3]], [arr_3[0], arr_3[3]]]
        for x, y in intervals:
            for z, w in intervals_2:
                if self.isintersect(*x, *y, *z, *w):
                    #print("INTERSECTION: ", x, y, z, w)
                    return True
        return False

    def getPoints(self):
        x, y, LL, h2, d, D = -self.y, self.x, self.L, self.h / 2, self.d, self.D
        truck_or, trailer_or = self.theta, self.theta + self.gamma
        
        truck = np.array([[-h2, D + d], [h2, D + d], [-h2, -d], [h2, -d]])
        trailer = np.array([[-h2, d], [h2, d], [-h2, -LL - d], [h2, -LL - d]])
        truck_orientation = np.vstack([truck @ np.array([np.cos(truck_or), -np.sin(truck_or)]), truck @ np.array([np.sin(truck_or), np.cos(truck_or)])]).T
        trailer_orientation = np.vstack([trailer @ np.array([np.cos(trailer_or), -np.sin(trailer_or)]), trailer @ np.array([np.sin(trailer_or), np.cos(trailer_or)])]).T
        return np.vstack([truck_orientation, trailer_orientation])



#print(t.getParams())
#t = Trailer()
#t.setParams(1., 1., 0., 0.)  
#print(t.getPoints())
#A = (-5, 0)
#B = (0, 0)
#C = (-5, -5)
#D = (3, 3)
#print(t.isintersect(*A, *B, *C, *D), "AXAXXAXAXAXAXA")



L, D = 6., 3.
v = 1.

A = np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., -1. / L]])
B = np.array([[0.], [1 / D], [-1 / D]])
Q = np.eye(3)
R = np.array([[1000.]])
#R_2 = np.array([[100.]])


P = solve_continuous_are(A, B, Q, R)
P_2 = solve_continuous_are(-A, -B, Q, R)
K_f = np.dot(np.linalg.inv(R), np.dot(np.transpose(B), P))
#K_f = np.array([[0.0316, .4367, 0.]])
K_b = np.dot(np.linalg.inv(R), np.dot(np.transpose(-B), P_2))
K_bb = K_b.copy()
#print(K_f)
#print(K_b)

def coords_for_u(currr, targg):
    xx = (currr[0] - targg[0]) * np.cos(targg[2]) + (currr[1] - targg[1]) * np.sin(targ[2])
    yy = (targg[0] - currr[0]) * np.sin(targg[2]) + (currr[1] - targg[1]) * np.cos(targ[2])
    ttheta = currr[2] - targg[2]
    ggamma = currr[-1]
    return np.array([xx, yy, ttheta, ggamma])


def vdp(t, vec, D, L, x_targ):
    x, y, theta, gamma = vec
    u = (-K_f @ coords_for_u(vec, x_targ)[1:])[0]
    return [np.cos(theta), np.sin(theta), u / D, -1/L * np.sin(gamma) - u / D]


def vdp_back(t, vec, D, L, x_targ):
    x, y, theta, gamma = vec[:4]
    XX = vec[1:4]
    e_t = e(t)[0]
    u = -(K_b[0] @ np.array([y, theta, gamma])) + e_t
    arr = np.array([-np.cos(theta), -np.sin(theta), -u / D, 1/L * np.sin(gamma) + u / D])
    sigma = np.array([theta - np.sin(theta), np.sin(gamma) - gamma])
    r_k = (XX.T @ (Q + R[0][0] * K_b.T @ K_b)) @ XX
    #print(r_k)
    #print(R[0][0] * e_t * XX, np.kron(e_t * R[0], XX))
    #return np.hstack([arr, np.kron(sigma, XX), np.kron(e_t * R[0], XX), np.array([r_k])])
    return np.hstack([arr, np.kron(sigma, XX), R[0][0] * e_t * XX, np.array([r_k])])

#print(solution.y.T)


def get_arr(trailer, size_scr, scale=10):
    x, y = trailer.getCoords()
    arr = t.getPoints() * scale
    arr[:, 1] = -arr[:, 1]
    X, Y = size_scr[0] // 2, size_scr[1] // 2
    arr[:, 0] += X + x * scale
    arr[:, 1] += Y - y * scale
    a1, a2, a3, a4, a5, a6, a7, a8 = arr
    return np.array([a1, a2, a4, a3, a5, a6, a8, a7])

def get_Borders(trailer, size_scr, scale=10):
    arr = trailer.getBorders() * scale
    arr[:, 1] = -arr[:, 1]
    X, Y = size_scr[0] // 2, size_scr[1] // 2
    arr[:, 0] += X
    arr[:, 1] += Y
    return arr

def get_Field(trailer, size_scr, scale=10):
    arr = trailer.getField() * scale
    arr[:, 1] = -arr[:, 1]
    X, Y = size_scr[0] // 2, size_scr[1] // 2
    arr[:, 0] += X
    arr[:, 1] += Y
    return arr

def draw_car(pg, sc, t, sz_sc, scale=10):
    arr = get_arr(t, sz_sc, scale)
    pg.draw.polygon(sc, GRAY, list(arr[:4]))
    pg.draw.polygon(sc, BLACK, arr[4:])


def draw_field(pg, sc, t, sz_sc, scale=10):
    arr = get_Borders(t, sz_sc, scale)
    arr_2 = get_Field(t, sz_sc, scale)
    pg.draw.line(sc, GRAY, arr[0], arr[1], 2)
    pg.draw.line(sc, GRAY, arr[2], arr[3], 2)
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


def exact_rank_sympy(matrix):
    M_sym = sp.Matrix(matrix.tolist())
    rank = M_sym.rank()
    return rank


#if __name__ == "__main__"
pygame.init()
size = 1200, 720
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Трейлер")
clock = pygame.time.Clock()
exit = False

pygame.display.flip()
#sol = solution.y.T
'''sol = np.array([[5., 0., 0., 0.], 
                [5., 0., 0., 0.],
                [4., 1., .1, 0.],
                [3., 2., .15, 0.],
                [2., 3., 0.18, 0.],
                [1., 4., 0.2, 0.],
                [0., 5., 0.25, 0.],
                [-1., 6., 0.28, 0.],
                [-2., 7., 0.3, 0.],
                [-3., 8., 0.3, 0.]])'''

Delta_A = np.array([[1., 0.], [0., 0.], [0., 1 / L]])
t = Trailer(L + 5, D)
t.setField(-15., 15., -15., 25.)
t.setBorders(-2., -15., -2., 4., 2., -15., 2., 4.)

curr = np.array([0., -5., .0, 0.])

#x3 = np.array([20., -5., -.3, 0.])
#x3 = np.array([20., 0., 0., 0.])
#x3 = np.array([20., 5., .3, 0.])
x3 = np.array([20., 2., .3, 0.])

#x2 = np.array([20., -5., -.3, 0.])
x2 = np.array([20., 0., 0., 0.])
#x2 = np.array([20., 5., .3, 0.])

#x1 = np.array([20., 5., .3, 0.])
x1 = np.array([20., 2., .3, 0.])

goal = np.zeros(4)

targ = x1

P_k_back = None
flag_for_P = False
nnn = 0
cnt = 0
time = 40
ccnt = 161
fps  = 24
while not exit:
    cnt += 1
    if cnt > 6:
        break
    sol = solve_ivp(vdp, [0., time], curr, t_eval=np.linspace(0., time, ccnt), args=(D, L, targ)).y.T
    for item in sol:
        t.setParams(*item)
        if t.check_stop():
            break
        curr = item.copy()
        screen.fill((255, 255, 255))
        draw_car(pygame, screen, t, size, 10)
        draw_field(pygame, screen, t, size, 10)
        clock.tick(fps)
        pygame.display.flip()
    soll = solve_ivp(vdp_back, [0., time], np.hstack([curr, np.zeros(10)]), t_eval=np.linspace(0., time, ccnt), args=(D, L, goal)).y
    sol_2 = soll[:4].T
    sol_3 = soll[4:].T
    Theta = list()
    Xi = list()
    nnn = 0
    flag_for_theta = False
    x_for_kron = sol_2[0][1:].copy()
    for index, item in enumerate(sol_2):
        t.setParams(*item)
        if t.check_stop() or abs(item[3]) > .9:
            break
        if flag_for_theta:
            Theta.append(np.hstack([get_v_for_dxx(item[1:]) - get_v_for_dxx(x_for_kron),
                                 -2 * (sol_3[index][:-1] - sol_3[index - 1][:-1])]))
            Xi.append(-(sol_3[index][-1] - sol_3[index - 1][-1]))
            x_for_kron = item[1:].copy()
        if index == 0:
            flag_for_theta = True
        curr = item.copy()
        screen.fill((255, 255, 255))
        draw_car(pygame, screen, t, size, 10)
        draw_field(pygame, screen, t, size, 10)
        clock.tick(fps)
        pygame.display.flip()
    np_Theta = np.array(Theta.copy())
    np_Xi = np.array(Xi.copy())
    nnn = np.linalg.matrix_rank(np_Theta)
    if nnn == 15:
        vect = np.linalg.pinv(np_Theta, rcond=None) @ np_Xi
        Pk = get_Pk(vect, 3)
        if not flag_for_P:
            flag_for_P = True
        else:
            print("Error of Pk: ", np.linalg.norm(Pk - P_k_back))
        print(Pk @ Delta_A)
        print(vect[6:-3])
        P_k_back = Pk.copy()
        K_b = np.array([vect[-3:]]).copy()
        print(K_b)
        #print("vect: ", K_b)
    Theta.clear()
    Xi.clear()
    #print(K_b)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit = True
            break
    curr_y = curr[1]
    if curr_y >= .3:
        targ = x1
    elif curr_y <= .3 and abs(curr_y) >= .05:
        targ = x2
    #elif abs(curr_y) > .5:
    #    targ = x2
    else:
        if curr[0] < 4.:
            exit = True
            break
#print(t.getPoints())
    #pygame.draw.polygon(screen, WHITE, )
exit = False
while not exit:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit = True
            break
print(cnt, "ИТЕРАЦИЙ!")
pygame.quit()