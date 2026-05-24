 # -*- coding: utf-8 -*-


from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
import sys
import os
import json
import time
import shutil
from datetime import datetime
import sympy as sp
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp, quad_vec
from scipy.linalg import solve_continuous_are
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.axes = self.figure.add_subplot(111)

        super(PlotCanvas, self).__init__(self.figure)
        self.setParent(parent)

    def plot(self, indexes):
        self.axes.clear()
        self.axes.set_title("Зависимость координат состояния " + "от t")
        self.axes.set_xlabel("Ось t")
        self.axes.set_ylabel("Ось xi(t)")
        y, t = self.parent().point, self.parent().t_point
        if len(indexes) == 0:
            indexes = list(range(len(y)))
        for index in indexes:
            self.axes.plot(t, y[index], label=f'x{index + 1}(t)', linewidth=2)
        self.axes.legend()
        self.figure.tight_layout()
        self.draw()


class Window(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__()
        uic.loadUi("ui/kursovaya_main.ui", self)
        self.setupUI()

    def setupUI(self):
        self.show_P_K_known_A_B = None
        self.solution = None
        self.n = 0
        self.m = 0
        self.q = 0
        self.spinBox.valueChanged.connect(self.__update_A_count)
        for table in [self.table_A, self.table_B, self.table_Q, self.table_R]:
            header = table.horizontalHeader()
            header.sectionDoubleClicked.connect(self.__resize_columns)
        self.load_action.triggered.connect(self.__load_system)
        self.save_action.triggered.connect(self.__save_system)
        self.__update_A_count(True)
        self.spinBox_4.valueChanged.connect(self.__update_B_count)
        self.__update_B_count(True)
        self.run_Rikkati.clicked.connect(self.__solve_Rikkati)
        self.__load_system(True)

    def __resize_columns(self):
        self.sender().parent().resizeColumnsToContents()

    def __update_A_count(self, start=False):
        n = self.spinBox.value()
        nn = self.n
        if (self.n < n) or (start is True):
            for _ in range(n - nn):
                for table in [self.table_A, self.table_B, self.table_Q]:
                    table.insertRow(self.n)
                    if table != self.table_B:
                        table.insertColumn(self.n)
                    for i in range(self.n + 1):
                        table.setItem(self.n, i, QTableWidgetItem("0"))
                        if (table != self.table_B) and (i != self.n):
                            table.setItem(i, self.n, QTableWidgetItem("0"))
                self.n += 1
        else:
            for _ in range(nn - n):
                for table in [self.table_A, self.table_B, self.table_Q]:
                    table.removeRow(self.n - 1)
                    if (table != self.table_B):
                        table.removeColumn(self.n - 1)            
                self.n -= 1

    def __update_B_count(self, start=False):
        m = self.spinBox_4.value()
        mm = self.m
        if (self.m < m) or (start is True):
            for _ in range(m - mm):
                for table in [self.table_B, self.table_R]:
                    table.insertColumn(self.m)
                    if (table == self.table_R):
                        table.insertRow(self.m)
                    if (table == self.table_B):
                        for i in range(self.n):
                            table.setItem(i, self.m, QTableWidgetItem("0"))
                    if (table == self.table_R):
                        for j in range(self.m + 1):
                            table.setItem(self.m, j, QTableWidgetItem("0"))
                            if (j != self.m):
                                table.setItem(j, self.m, QTableWidgetItem("0"))
                self.m += 1
        else:
            for _ in range(mm - m):
                for table in [self.table_B, self.table_R]:
                    table.removeColumn(self.m - 1)
                    if (table == self.table_R):
                        table.removeRow(self.m - 1)
                self.m -= 1

    def get_matrix(self, *tables):
        return tuple(np.array([[float(table.item(i, j).text()) for j in range(table.columnCount())] for i in range(table.rowCount())]) for table in tables)

    def __solve_Rikkati(self):
        try:
            A, B, Q, R = self.get_matrix(self.table_A, self.table_B, self.table_Q, self.table_R)
            AA, BB, C = A, np.dot(B, np.dot(np.linalg.inv(R), np.transpose(B))), Q
            P = solve_continuous_are(A, B, Q, R)
            K = np.dot(np.linalg.inv(R), np.dot(np.transpose(B), P))
            self.show_P_K_known_A_B = P_K_known_A_B(self, P, K)
            self.show_P_K_known_A_B.show()
        except Exception as message:
            print("Error: ", message)

    def __save_system(self):
        A, B, Q, R = self.get_matrix(self.table_A, self.table_B, self.table_Q, self.table_R)
        st = str(datetime.now())
        st = st[:st.find('.')].replace(':', '.')
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить", f"user_files/{st}.npz", "NPZ Files (*.npz)")
        if path:
            np.savez(path, A=A, B=B, Q=Q, R=R)
            self.statusbar.showMessage(f"Система была сохранена в файл: {path}")

    def __load_system(self, for_run=False):
        if for_run:
            answer = "user_files/default.npz"
        else:
            answer, _ = QFileDialog.getOpenFileName(self, "Выбор файла", "user_files", "NPZ Files (*.npz)")
        if answer:
            try:
                data = np.load(answer)
                arrays = [data['A'], data['B'], data['Q'], data['R']]
                self.spinBox.setValue(arrays[0].shape[0])
                self.spinBox_4.setValue(arrays[1].shape[1])
                for i, table in enumerate([self.table_A, self.table_B, self.table_Q, self.table_R]):
                    table.clear()
                    self.__fill_table(table, arrays[i])
                    table.resizeColumnsToContents()
                self.statusbar.showMessage(f"Система из файла {answer} успешно загружена")
            except Exception as m:
                print(m)
                self.statusbar.showMessage(f"Не удалось обработать файл: {answer}")

    def __fill_table(self, table, arr):
        x, y = arr.shape
        for i in range(x):
            for j in range(y):
                table.setItem(i, j, QTableWidgetItem(str(arr[i][j])))

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Close):
            self.close()
        elif event.matches(QKeySequence.Open):
            self.__load_system()
        elif event.matches(QKeySequence.Save):
            self.__save_system()


class P_K_known_A_B(QWidget):
    def __init__(self, parent, *args, **kwargs):
        super().__init__()
        uic.loadUi("ui/show_P_K_with_known_A_B.ui", self)
        self.setupUI(parent, *args)

    def setupUI(self, parent, *args):
        self.parent = parent
        P, K = args
        self.K = K
        self.run_cool_algoritm = None
        self.run_algoritm.clicked.connect(self.__run_cool_algoritm)
        self.parent.setDisabled(True)
        self.P_arr, self.K_arr = args
        self.__fill_table(self.table_P, P)
        self.__fill_table(self.table_K, K)

    def __run_cool_algoritm(self):
        self.run_cool_algoritm = P_K_without_A_B(self.parent, self, self.parent.table_B.rowCount(), self.parent.table_B.columnCount())
        self.run_cool_algoritm.show()

    def closeEvent(self, event):
        self.parent.setEnabled(True)

    def __fill_table(self, table, arr):
        x, y = arr.shape
        table.setRowCount(x)
        table.setColumnCount(y)
        for i in range(x):
            for j in range(y):
                item = QTableWidgetItem(f"{arr[i][j]:.4f}")
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table.setItem(i, j, item)
                table.resizeColumnsToContents()

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Close):
            self.close()


class P_K_without_A_B(QWidget):
    def __init__(self, grandparent, parent, *args, **kwargs):
        super().__init__()
        uic.loadUi("ui/show_P_K_without_A_B_full.ui", self)
        self.setupUI(grandparent, parent, *args)

    def setupUI(self, grandparent, parent, *args):
        self.Pk_pred = None
        n, m = args
        self.n, self.m = n, m
        self.iteration_count = 0
        self.Ixx, self.Ixu, self.dxx, self.Kk, self.R, self.Q, self.Pk = None, None, None, None, None, None, None
        self.P = self.get_table(parent.table_P)
        self.__update_table(self.table_K0, m, n)
        #self.__update_table(self.table_e, m, 1, arr=[["e" + str(i + 1)] for i in range(m)], editable=False)
        self.__update_table(self.table_P_k, n, n, editable=False)
        self.__update_table(self.table_K_k, m, n, editable=False)
        self.grandparent, self.parent = grandparent, parent
        #_______________for_run____________
        self.run.clicked.connect(self.__for_run)
        self.exit.clicked.connect(self.close)
        self.break_run.clicked.connect(self.__for_break_run)
        self.next_step.clicked.connect(self.__run_cycle)

        self.point = None
        self.t_point = None
        self.x0 = None
        
        self.canvas = PlotCanvas(self)
        self.arr_of_check = list()
        lay = QVBoxLayout()
        for i in range(n):
            item = QCheckBox("x" + str(i + 1) + "(t)", self)
            item.toggled.connect(self.__run_graph)
            lay.addWidget(item)
            self.arr_of_check.append(item)
        self.layout().addLayout(lay)
        self.layout().addWidget(self.canvas)
        self.__is_run_graph = False

    def __update_table(self, table, n=None, m = None, arr=None, editable=True):
        if not n:
            n = self.table.rowCount()
        if not m:
            m = self.table.columnCount()
        table.clear()
        table.setRowCount(n)
        table.setColumnCount(m)
        if arr is None:
            arr = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                item = QTableWidgetItem(f"{arr[i][j]:.4f}")
                if not editable:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                else:
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                table.setItem(i, j, item)
        table.resizeColumnsToContents()

    def get_table(self, table):
        n, m = table.rowCount(), table.columnCount()
        return np.array([[float(table.item(i, j).text()) for j in range(m)] for i in range(n)])

    def __for_run(self):
        self.iteration_count = 0
        self.iteration.display(0)        
        A = self.get_table(self.grandparent.table_A)
        B = self.get_table(self.grandparent.table_B)
        Q = self.get_table(self.grandparent.table_Q)
        R = self.get_table(self.grandparent.table_R)
        K0 = self.get_table(self.table_K0)

        eigenvalues = np.linalg.eigvals(A - B @ K0)
        if not np.all(np.real(eigenvalues) < 0):
            QMessageBox.information(self, "Невыполнение условия", "Внимание, матрица K0 не является стабилизирующей!")
            return


        t = 0.
        step = float(self.step.value())
        super_step = self.super_step.value()
        cycle_count = self.cycle.value()

        dxx, xu, xx = list(), list(), list()
        n, m = self.n, self.m

        e = ExplorationNoise(m)

        x0 = np.random.randn(self.n)
        self.point = list([x0])
        self.t_point = list([t])
        t_grid = np.linspace(0, t + step * cycle_count, cycle_count + 1)
        e_grid = np.array([e(tt) for tt in t_grid])
        e_interp = interp1d(t_grid, e_grid, kind='linear', axis=0, fill_value='extrapolate', assume_sorted=True)
        ee = lambda tttt: - K0 @ x0 if tttt == 0. else e_interp(tttt)


        nnn_count = n * (n + 1) // 2 + m * n
        self.lll_count.setText(str(nnn_count))
        count = 0


        def func(t, x):
            u = -K0 @ x + ee(t)
            return A @ x + B @ u


        dxx_pred = get_v_for_dxx(x0)

        x_interp = lambda _: None

        def integrand_xx(t):
            x_val = x_interp(t)
            return np.kron(x_val, x_val)
        
        
        def integrand_xu(t):
            x_val = x_interp(t)
            u_val = -K0 @ x_val + ee(t)
            return np.kron(x_val, u_val)        

        for i in range(cycle_count):
            print(i)
            t_span = (t, t + step)
            t_eval = np.linspace(t, t + step, super_step + 1)
            sol = solve_ivp(func, t_span, x0, t_eval=t_eval, method='DOP853', rtol=1e-12, atol=1e-14)
            solution = sol.y.T
            dxx_curr = get_v_for_dxx(solution[-1])
            dxx.append(dxx_curr - dxx_pred)
            dxx_pred = dxx_curr

            x_interp = interp1d(sol.t, sol.y, kind='cubic', axis=1)


            Ixx, _ = quad_vec(integrand_xx, t, t + step, epsabs=1e-13, epsrel=1e-13)
            Ixu, _ = quad_vec(integrand_xu, t, t + step, epsabs=1e-13, epsrel=1e-13)

            xx.append(Ixx)
            xu.append(Ixu)

            t += step
            x0 = solution[-1]
            self.point.append(x0)
            self.t_point.append(t)
        count = exact_rank_sympy(np.hstack([np.array(xx), np.array(xu)]))
        self.rang_count.setText(str(count))
        nnn_count = count
        if count == n * (n + 1) // 2 + m * n:
            self.T.setValue(t)
            self.run.setDisabled(True)
            self.stop_in_step.setDisabled(True)
            self.__update_table(self.table_K_k, m, n, editable=False)
            self.presicion.setDisabled(True)
            self.cycle.setDisabled(True)
            self.Kk = K0.copy()
            self.R = R.copy()
            self.Q = Q.copy()
            self.dxx = np.array(dxx)
            self.Ixx = np.array(xx)
            self.Ixu = np.array(xu)
            if self.stop_in_step.isChecked():
                self.next_step.setEnabled(True)
                self.break_run.setEnabled(True)
                self.__run_cycle()
            else:
                k_pred = 0
                while not self.presicion.isEnabled():
                    k_pred += 1
                    self.__run_cycle()
                    if k_pred > 1000:
                        QMessageBox.information(self, "Бесконечный цикл", "Внимание, программа улетела в закат!")
                        self.__for_break_run()
                        break
                self.x0 = x0
                self.__is_run_graph = True
                self.__show_graph()
        else:
            QMessageBox.information(self, "Вычисление матриц bxx, Ixx, Ixu", "Внимание! Не выполнено условие Леммы!")

    def __run_graph(self):
        if self.__is_run_graph:
            self.canvas.plot(self.__get_indexes())

    def __show_graph(self):
        A = self.get_table(self.grandparent.table_A)
        B = self.get_table(self.grandparent.table_B)
        Kk = self.get_table(self.table_K_k)
        def func(t, x):
            u = -Kk @ x
            return A @ x + B @ u
        t = self.t_point[-1]
        solution = solve_ivp(func, (t, t + 10), self.x0, t_eval=np.linspace(t, t + 10, 11))
        self.point = np.hstack([np.array(self.point).T, solution.y])
        self.t_point = np.hstack([np.array(self.t_point), solution.t])
        self.__run_graph()

    def __run_cycle(self):
        Theta = np.hstack([self.dxx, -2. * (self.Ixx @ np.kron(np.eye(self.n), (self.Kk.T @ self.R)) + (self.Ixu @ np.kron(np.eye(self.n), self.R)))])
        Qk = self.Q + (self.Kk.T @ self.R @ self.Kk)
        Xi = -self.Ixx @ Qk.flatten(order='F')
        vec = np.linalg.pinv(Theta, rcond=None) @ Xi
        index = self.m * self.n
        Kk1 = np.transpose(np.array(vec[-index:]).reshape((self.n, self.m)))
        Pk = get_Pk(vec, self.n)
        self.__update_table(self.table_P_k, self.n, self.n, Pk, False)
        self.__update_table(self.table_K_k, self.m, self.n, self.Kk, False)
        self.iteration_count += 1
        self.iteration.display(self.iteration_count)
        self.Kk = Kk1.copy()
        if self.iteration_count == 1:
            self.Pk = Pk.copy()
            self.Pk_pred = Pk.copy()
        else:
            self.Pk_pred = self.Pk
            self.Pk = Pk.copy()
            if np.linalg.norm(self.Pk_pred - self.Pk) <= self.presicion.value():
                self.__for_break_run()

    def __for_break_run(self):
        self.run.setEnabled(True)
        self.next_step.setDisabled(True)
        self.break_run.setDisabled(True)
        self.stop_in_step.setEnabled(True)
        self.presicion.setEnabled(True)
        self.cycle.setEnabled(True)
        self.run.setEnabled(True)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Close):
            self.close()
        elif event.matches(QKeySequence.Save):
            self.__open_save_system()

    def __open_save_system(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить график", get_random_path(),"Изображения (*.png *.jpg *.jpeg *.bmp *.svg *.pdf);;Все файлы (*)")
        if file_path:
            try:
                self.canvas.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Сохранение графика", f"График сохранен как: {file_path}")
            except Exception as e:
                QMessageBox.information(self, "Сохранение графика", "Ошибка сохранения")        

    def __get_indexes(self):
        return list([i for i, item in enumerate(self.arr_of_check) if item.isChecked()])


class ExplorationNoise:
    def __init__(self, m, n_frequencies=100, amplitude=100, freq_range=500):
        self.m = m
        self.amplitude = amplitude
        self.omega = np.random.uniform(-freq_range, freq_range, (m, n_frequencies))
    
    def __call__(self, t):
        return self.amplitude * np.sum(np.sin(self.omega * t), axis=1)


def get_v_for_dxx(vec):
    arr, n = [], len(vec)
    for i in range(n):
        for j in range(i, n):
            arr.append(vec[i] * vec[j])
    return np.array(arr)


def get_Pk(vector, n):
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


def check_cond_lemma(Ixx, Ixu):
    M = np.hstack([Ixx, Ixu])
    return np.linalg.matrix_rank(M, rtol=1e-10)


def exact_rank_sympy(matrix):
    M_sym = sp.Matrix(matrix.tolist())
    rank = M_sym.rank()
    return rank


def get_random_path():
    st = str(datetime.now())
    return st[:st.find('.')].replace(':', '.')


# catch errors
def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == "__main__":
    # show interface
    app = QApplication(sys.argv)
    form = Window()
    form.show()
    sys.excepthook = except_hook
    sys.exit(app.exec())