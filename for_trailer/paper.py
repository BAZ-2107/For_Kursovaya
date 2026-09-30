# -*- coding: utf-8 -*-


from scipy.integrate import solve_ivp
import numpy as np
from scipy.linalg import solve_continuous_are
import pygame



GRAY = (125, 125, 125)
BLACK=(0, 0, 0)


class ExplorationNoise:
    def __init__(self, m, n_frequencies=100, amplitude=100, freq_range=500):
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
        D = k1 * k4 - k3 * k2
        if D != 0.:
            k5, k6 = c2 - a2, d2 - b2
            return (0 <= ((k1 * k6 - k3 * k5) / D) <= 1) and (0 <= ((k5 * k4 - k2 * k6) / D) <= 1)
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
        x, y, L, h2, d, D = -self.y, self.x, self.L, self.h / 2, self.d, self.D
        truck_or, trailer_or = self.theta, self.theta + self.gamma
        
        truck = np.array([[-h2, D + d], [h2, D + d], [-h2, -d], [h2, -d]])
        trailer = np.array([[-h2, d], [h2, d], [-h2, -L - d], [h2, -L - d]])
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

A = np.array([[0., v, 0.], [0., 0., 0.], [0., 0., -v / L]])
B = np.array([[0.], [v / D], [-v / D]])
Q = np.eye(3)
R = np.array([[1000.]])


P = solve_continuous_are(A, B, Q, R)
P_2 = solve_continuous_are(-A, -B, Q, R)
K_f = np.dot(np.linalg.inv(R), np.dot(np.transpose(B), P))
#K_b = np.dot(np.linalg.inv(R), np.dot(np.transpose(-B), P_2))
K_b = np.array([[-.1, 2.9843, 4.3163]])
#print(K_f)
#print(K_b)

def coords_for_u(curr, targ):
    xx = (curr[0] - targ[0]) * np.cos(targ[2]) + (curr[1] - targ[1]) * np.sin(targ[2])
    yy = (targ[0] - curr[0]) * np.sin(targ[2]) + (curr[1] - targ[1]) * np.cos(targ[2])
    ttheta = curr[2] - targ[2]
    ggamma = curr[-1]
    return np.array([xx, yy, ttheta, ggamma])


def vdp(t, vec, D, L, x_targ):
    x, y, theta, gamma = vec
    u = (-K_f @ coords_for_u(vec, x_targ)[1:])[0]
    return [v * np.cos(theta), v * np.sin(theta), v * u / D, -v/L * np.sin(gamma) - v * u / D]


def vdp_back(t, vec, D, L, x_targ):
    x, y, theta, gamma = vec
    u = (-K_b @ coords_for_u(vec, x_targ)[1:])[0]
    arr = np.array([-v * np.cos(theta), -v * np.sin(theta), -v * u / D, v/L * np.sin(gamma) + v * u / D])
    
    return arr

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

t = Trailer(L + 5, D)
t.setField(-15., 15., -15., 25.)
t.setBorders(-2., -15., -2., 4., 2., -15., 2., 4.)

curr = np.array([0., -5., .0, 0.])

x3 = np.array([20., -5., -.7, 0.])

x2 = np.array([20., -5., -.3, 0.])

x1 = np.array([20., 5., .3, 0.])

goal = np.array([0., 0., 0., 0.])

targ = x1

cnt = 0
time = 80
ccnt = 81
fps  = 24
while not exit:
    cnt += 1
    if cnt > 15:
        break
    sol = solve_ivp(vdp, [0., time], curr, t_eval=np.linspace(0., time, ccnt), args=(D, L, targ)).y.T
    for item in sol:
        t.setParams(*item)
        if t.check_stop():# or curr[0] > targ[0]:
            break
        curr = item
        screen.fill((255, 255, 255))
        draw_car(pygame, screen, t, size, 10)
        draw_field(pygame, screen, t, size, 10)
        clock.tick(fps)
        pygame.display.flip()
    soll = solve_ivp(vdp_back, [0., time], curr, t_eval=np.linspace(0., time, ccnt), args=(D, L, goal)).y.T
    for item in soll:
        t.setParams(*item)
        if abs(item[3]) > .9:
            print("YES")
        if t.check_stop():
            break
        curr = item
        screen.fill((255, 255, 255))
        draw_car(pygame, screen, t, size, 10)
        draw_field(pygame, screen, t, size, 10)
        clock.tick(fps)
        pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit = True
            break
    curr_y = curr[1]
    if curr_y > 1.2:
        targ = x1
    elif curr_y < -1.2:
        targ = x3
    elif abs(curr_y) > .5:
        targ = x2
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