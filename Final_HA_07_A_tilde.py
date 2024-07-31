# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:08:09 2024

@author: carol
"""

import numpy as np
import scipy.sparse
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
import time
'''
Aufgabe 1, Initialisierung
'''

'''
Aufgabe 2, nützliche Arrays
'''


# a) 3D-Arrays
def getindizes(parameters: object):
    # a) 3D-Arrays
    nv = np.arange(0, 4, 1)  # Create an array with values [0, 1, 2, 3]
    J, I = np.meshgrid(nv, nv)  # repeats the rows, columns of nv in J, I

    matl = np.arange(parameters.n).reshape(parameters.n, 1, 1) * np.ones((1, 4, 4)).astype(
        int)  # Create a 3D array, where each element from 0 to parameters.n-1 is repeated in shape(4,4)
    mati = np.repeat(I[np.newaxis, :, :], parameters.n,
                     axis=0)  # Repeat the  array I along the first axis parameters.n times
    matj = np.repeat(J[np.newaxis, :, :], parameters.n,
                     axis=0)  # Repeat the  array J along the first axis parameters.n times

    matlli = (2 * matl + mati).astype(int)  # Calculate a 3D array from  matl & mati
    matllj = (2 * matl + matj).astype(int)  # Calculate a 3D array from  matl & matj

    # We decided to use a row vector to represent the vector here
    # You can use a column vector as well [[[0], [0], [0], [0]], [[1], [1], ..
    # It does not matter that much as long as we stay consistent with this vector (matrix)
    # We use sparce matrices, so we flatten the vector anyway!
    veki, vekl = np.meshgrid(nv, np.arange(0,
                                           parameters.n))  # Create 2D  arrays  using nv and an array from 0 to parameters.n-1
    veklli = (2 * vekl + veki).astype(int)  # Calculate a 2D array from  vekl + veki

    # Return the generated arrays
    return matl, mati, matj, matlli, matllj, vekl, veki, veklli


'''
Aufgabe 3,4; Elementmatrizen, -vektoren
'''


# create Mass matrix
def getMbar(parameters: object):
    h = parameters.h
    faktor = parameters.my(x) * h / 420  # define factor
    matrix = np.array(
        [[156, 22 * h, 54, -13 * h], [22 * h, 4 * h ** 2, 13 * h, -3 * h ** 2], [54, 13 * h, 156, -22 * h],
         [-13 * h, -3 * h ** 2, -22 * h, 4 * h ** 2]])  # define matrix
    M = faktor * matrix  # scale matrix by the factor
    M = np.tile(M, (parameters.n, 1, 1))  # Replicate the scaled matrix for each element
    # Return the final 3D array
    return M


# create Stiffness matrix S the same way as M in getMbar
def getSbar(parameters: object):
    h = parameters.h
    faktor = parameters.E(x) * parameters.I(x) / h ** 3
    matrix = np.array([[12, 6 * h, -12, 6 * h], [6 * h, 4 * h ** 2, -6 * h, 2 * h ** 2], [-12, -6 * h, 12, -6 * h],
                       [6 * h, 2 * h ** 2, -6 * h, 4 * h ** 2]])
    S = faktor * matrix
    S = np.tile(S, (parameters.n, 1, 1))
    return S


# create element vector vekq the same way as M in getMbar
def getqbar(parameters: object):
    h = parameters.h
    faktor = parameters.q(x) * h / 12
    vektor = np.array([[6], [h], [6], [-h]])
    vekq = faktor * vektor
    vekq = np.tile(vekq, (parameters.n, 1, 1))
    return vekq


'''
Aufgabe 5, Massen-, Steifigkeitsmatrix, Streckenlastvektor
'''


# Mass matrix
def getM(parameters: object, indexes: object):
    M_alt = getMbar(parameters)  # Define the data matrix
    M_neu = coo_matrix((M_alt.flatten(), (indexes.matlli.flatten(), indexes.matllj.flatten()))).tocsr()
    return M_neu


# Stiffness matrix
# Analogous to getM for the data of the stiffness matrix
def getS(parameters, indexes):
    S_alt = getSbar(parameters)
    S_neu = coo_matrix((S_alt.flatten(), (indexes.matlli.flatten(),
                                          indexes.matllj.flatten()))).tocsr()  # Convert the data matrix to a sparse matrix in COO format and then to CSR format
    return S_neu


# element vector
# Analogous to getM for the values of the element vector
def getvq(parameters, indexes):
    vq_alt = getqbar(parameters)
    vq_neu = coo_matrix((vq_alt.flatten(), (indexes.veklli.flatten(), np.zeros_like(indexes.veklli.flatten())))).tocsr()
    # np.zeros_like(veklli.flatten()) creates an array of zeros with the same shape as veklli.flatten()
    return vq_neu


'''
Aufgabe 6
'''


# Sind und nicht ganz sicher ob wir "==" verwenden dürfen, deshalb gubt es zwei Varianten
# B is not getting passed into the function, so we have to be careful when we redefine the B matrix!!

def getC(parameters):
    E1_indices = parameters.B[parameters.B[:, 1] == 1, 0]
    E2_indices = parameters.B[parameters.B[:, 1] == 2, 0]

    C1_indices = np.concatenate((E1_indices * 2, E2_indices * 2 + 1))

    assert max(C1_indices) <= parameters.n, "Update the B matrix for current n"  # This does not catch all errors!

    num_entries = len(C1_indices)
    C1 = coo_matrix((np.ones(num_entries), (C1_indices, np.arange(num_entries))),
                    shape=(2 * parameters.n + 2, num_entries)).tocsr()

    return C1


'''
Aufgabe 7
'''


# Variante 1. Das ist wieder die Variante mit dem ==, man kann das aber genau so auch alternativ ohne machen
# The B matrix has to be updated to the current n before you use this function

def getvn(parameters):
    E3_indices = parameters.B[parameters.B[:, 1] == 3, 0]  # get all elements for K3, and extract the indices
    E4_indices = parameters.B[parameters.B[:, 1] == 4, 0]

    c_3 = parameters.B[parameters.B[:, 1] == 3, 2]  # get all elementd for K3 and extract the values to the indices
    c_4 = parameters.B[parameters.B[:, 1] == 4, 2]

    c_3_values = np.ones(len(E3_indices)) * c_3.T
    c_4_values = np.ones(len(E4_indices)) * c_4.T

    v_N_rows = np.concatenate((E3_indices, E4_indices)).astype(int)

    assert max(v_N_rows) <= parameters.n, "Update the B matrix for current n"  # check for update errors

    v_N_cols = np.zeros(len(v_N_rows)).astype(int)
    v_N_vals = np.concatenate((c_3_values, c_4_values))
    v_N_shape = (2 * parameters.n + 2, 1)

    v_N = coo_matrix((v_N_vals, (v_N_rows, v_N_cols)), v_N_shape).tocsr()

    return v_N


'''
Aufgabe 8
'''


def getvd(parameters):
    a_k_values = parameters.B[parameters.B[:, 1] == 1, 2]
    b_k_values = parameters.B[parameters.B[:, 1] == 2, 2]

    v_D_values = np.concatenate((a_k_values, b_k_values)).astype(int)
    v_D_rows = np.arange(len(v_D_values)).astype(int)
    v_D_cols = np.zeros(len(v_D_values)).astype(int)

    v_D = coo_matrix((v_D_values, (v_D_rows, v_D_cols))).tocsr()

    return v_D


'''
Aufgabe 9
'''


def getMe(parameters, indexes):
    M = getM(parameters, indexes)
    C = getC(parameters)
    C0 = np.zeros_like(C.toarray())
    zero_filler = scipy.sparse.csr_matrix(np.zeros((C0.shape[1], C0.shape[1])))
    M_C0_horizontal_stack = scipy.sparse.hstack([M, C0])
    filler_C0_horizontal_stack = scipy.sparse.hstack([C0.T, zero_filler])

    Me = scipy.sparse.vstack([M_C0_horizontal_stack, filler_C0_horizontal_stack])
    return Me


def getSe(parameters, indexes):
    S = getS(parameters, indexes)
    C = getC(parameters)
    zero_filler = scipy.sparse.csr_matrix(np.zeros((C.toarray().shape[1], C.toarray().shape[1])))

    S_C_horizontal_stack = scipy.sparse.hstack([S, C])
    C_filler_horizontal_stack = scipy.sparse.hstack([C.T, zero_filler])
    Se = scipy.sparse.vstack([S_C_horizontal_stack, C_filler_horizontal_stack]).tocsr()
    return Se


def getve(parameters, indexes):
    vq_nq = getvq(parameters, indexes)
    v_E = scipy.sparse.vstack([vq_nq, getvd(parameters)])
    return v_E


'''
Aufgabe 12, 14
'''


def newmark_simmulation(parameters, static_solution):
    beta = parameters.beta
    gamma = parameters.gamma
    eta = parameters.eta

    # Initialize starting point
    # use the static state as our startingpoint for the Newmark-algorithim

    a_0_rows = np.arange(len(static_solution), dtype=int)
    a_0_cols = np.zeros_like(static_solution, dtype=int)

    a_p = coo_matrix((static_solution, (a_0_rows, a_0_cols))).tocsr()  # deflections

    # In the static case there is no acceleration and initial velocity
    a_d_p = coo_matrix((np.zeros_like(static_solution), (a_0_rows, a_0_cols))).tocsr()  # velocities = 0
    a_dd_p = coo_matrix((np.zeros_like(static_solution), (a_0_rows, a_0_cols))).tocsr()  # acceleration = 0

    # prepare an empty matrix for coming animation data
    a_p_animation = np.zeros((parameters.n_p, 2 * parameters.n + 2))  # Data Matrix for the Animation
    total_energy_timesteps = np.zeros(parameters.n_p)  # Task 14, prepare vector

    # Iterate over n_p timesteps using the Newmark-Algorithm
    for time_step in range(parameters.n_p):
        # Explicit step for displacement and velocity
        a_explicit = a_p + (a_d_p * eta) + (0.5 - beta) * (a_dd_p * eta ** 2)
        a_d_explicit = a_d_p + (1 - gamma) * (a_dd_p * eta)
        # next Acceleration
        a_dd_p = scipy.sparse.linalg.spsolve(M_e + (S_e * beta * eta ** 2), v_e - S_e.dot(a_explicit))
        a_dd_p = coo_matrix((a_dd_p, (a_0_rows, a_0_cols))).tocsr()
        # next Displacement
        a_p = a_explicit + beta * a_dd_p * eta ** 2
        # fill in data in data matrix for animation
        a_p_animation[time_step, :] = a_p.toarray()[
                                      :2 * parameters.n + 2].T  # only the first 2n+2 elements are coordinates
        # next Velocity
        a_d_p = a_d_explicit + gamma * a_dd_p * eta

        # task 14
        loads_v = a_p[2 * parameters.n + 2:] * (-1)

        total_energy = 0.5 * a_d_p[:2 * parameters.n + 2].T @ M @ a_d_p[:2 * parameters.n + 2] + (
                0.5 * S * a_p[:2 * parameters.n + 2] - v_q - C @ loads_v - v_n).T @ a_p[:2 * parameters.n + 2]
        total_energy_timesteps[time_step] = total_energy[0].toarray().item()

    return a_p_animation, total_energy_timesteps


def getplot(a_p_animation, parameters):
    # Animation
    data = a_p_animation

    # Number of timesteps and supports
    num_time_steps = data.shape[0]

    # Create the X-axis (positions of the supports)
    x_data = np.linspace(0, parameters.l, len(a_p_animation[0]))

    # Initialize the figure and axes
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(data.min(), data.max())
    ax.set_xlabel("Balkenlänge")
    ax.set_ylabel("Verformung")
    ax.set_title("Biegung des Balkens über die Zeit")

    # Function to initialize the animation
    def init():
        line.set_data([], [])
        return line,

    # Update function for each frame of the animation
    def update(frame):
        y_data = a_p_animation[frame,
                 :]  # y_data = a_p_animation[frame]  # Only plot the deviations, not the forces of the alpha vector
        line.set_data(x_data, y_data)
        return line,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_time_steps, init_func=init, blit=True)

    # Save the animation as a GIF
    ani.save("balkenbiegung_ani.gif", writer=PillowWriter(fps=10))

    # Optional: Show the animation
    plt.show()
    plt.close()


'''
Aufgabe 13
'''


# calculate matrix A
def getA(parameters: object, indexes: object):
    h = 1  # The A-Matrix is onyl equal to the M matrix when h = 1
    faktor = parameters.my(x) * h / 420  # define factor
    matrix = np.array(
        [[156, 22 * h, 54, -13 * h], [22 * h, 4 * h ** 2, 13 * h, -3 * h ** 2], [54, 13 * h, 156, -22 * h],
         [-13 * h, -3 * h ** 2, -22 * h, 4 * h ** 2]])  # define matrix
    A = faktor * matrix  # scale matrix by the factor
    A = np.tile(A, (parameters.n, 1, 1))  # Replicate the scaled matrix for each element
    A_neu = coo_matrix((A.flatten(), (indexes.matlli.flatten(), indexes.matllj.flatten()))).tocsr()
    return A_neu


# calculate the analytic solution for the given problem
def analytical_w(parameters, x_k):
    return (parameters.q(x) / (parameters.E(x) * parameters.I(x))) * (
            (x_k ** 4) / 24 - (parameters.l * x_k ** 3) / 6 + ((parameters.l ** 2) * (x_k ** 2)) / 4)


def analytical_w_prime(parameters, x_k):
    return (parameters.q(x) / (parameters.E(x) * parameters.I(x))) * (
            (x_k ** 3) / 6 - (parameters.l * x_k ** 2) / 2 + (parameters.l ** 2 * x_k) / 2)


'''
Aufgabe 16
'''


def getstencil(parameters):
    k_tilde = parameters.ns + 1  # create number of supporting points
    x_k_tilde = np.linspace(0, parameters.l, parameters.ns + 1)  # create supporting points
    # create vandermonde matrix
    X, vandermonde = np.meshgrid(x_k_tilde, x_k_tilde)  # vandermonde without powers
    powers = np.arange(0, k_tilde, 1)  # create powers for vandermondematrix
    vanderm = vandermonde ** powers  # create final vandermonde matrix

    # calculate integrated vamdermonde matrix
    numero = np.ones(len(vanderm[0]))
    divider = np.arange(0, len(vanderm[0]), 1)
    vanderm_integ = numero / (divider + 1)

    # calculate stencil
    stencil = np.linalg.solve(vanderm.T, vanderm_integ)

    # return stencil
    return stencil


'''
Aufgabe 17, 18
'''


# a)

def getphi(parameters, indexes, quadrature_points):
    # x = quadrature points = Stützstellen
    # create Vektor of suoporting points
    # quadrature_points = np.linspace(0, parameters.l, parameters.ns + 1)

    # initialisiere leere matrix
    phi_i_lijk = np.zeros((parameters.n, 4, 4, len(quadrature_points)))
    phi_j_lijk = np.zeros((parameters.n, 4, 4, len(quadrature_points)))
    phi_i_lik = np.zeros((parameters.n, 4, len(quadrature_points)))

    # initialisiere Formfunktionen
    phi_0 = lambda x: 1 - 3 * x ** 2 + 2 * x ** 3
    phi_1 = lambda x: x - 2 * x ** 2 + x ** 3
    phi_2 = lambda x: 3 * x ** 2 - 2 * x ** 3
    phi_3 = lambda x: -x ** 2 + x ** 3

    # Weist die Formfunktionen basierend auf den Indexwerten zu
    phi_i_lik[indexes.veki == 0] = phi_0(quadrature_points)
    phi_i_lik[indexes.veki == 1] = phi_1(quadrature_points)
    phi_i_lik[indexes.veki == 2] = phi_2(quadrature_points)
    phi_i_lik[indexes.veki == 3] = phi_3(quadrature_points)

    phi_i_lijk[indexes.mati == 0] = phi_0(quadrature_points)
    phi_i_lijk[indexes.mati == 1] = phi_1(quadrature_points)
    phi_i_lijk[indexes.mati == 2] = phi_2(quadrature_points)
    phi_i_lijk[indexes.mati == 3] = phi_3(quadrature_points)

    phi_j_lijk[indexes.matj == 0] = phi_0(quadrature_points)
    phi_j_lijk[indexes.matj == 1] = phi_1(quadrature_points)
    phi_j_lijk[indexes.matj == 2] = phi_2(quadrature_points)
    phi_j_lijk[indexes.matj == 3] = phi_3(quadrature_points)

    phi = np.array(
        [phi_0(quadrature_points), phi_1(quadrature_points), phi_2(quadrature_points), phi_3(quadrature_points)])

    return phi_i_lijk, phi_j_lijk, phi_i_lik, phi


# b)
def getddphi(parameters, indexes):
    # create Vektor of suoporting points
    quadrature_points = np.linspace(0, parameters.l, parameters.ns + 1)

    # initialisiere Formfunktionen
    ddphi_0 = lambda x: -6 + 12 * x
    ddphi_1 = lambda x: -4 + 6 * x
    ddphi_2 = lambda x: 6 - 12 * x
    ddphi_3 = lambda x: -2 + 6 * x

    # Initialisiert leere Arrays für die zweiten Ableitungen der Formfunktionen
    ddphi_i_lijk = np.zeros((parameters.n, 4, 4, parameters.ns + 1))
    ddphi_j_lijk = np.zeros((parameters.n, 4, 4, parameters.ns + 1))

    # Weist die zweiten Ableitungen basierend auf den Indexwerten zu
    ddphi_i_lijk[indexes.mati == 0] = ddphi_0(quadrature_points)
    ddphi_i_lijk[indexes.mati == 1] = ddphi_1(quadrature_points)
    ddphi_i_lijk[indexes.mati == 2] = ddphi_2(quadrature_points)
    ddphi_i_lijk[indexes.mati == 3] = ddphi_3(quadrature_points)

    ddphi_j_lijk[indexes.matj == 0] = ddphi_0(quadrature_points)
    ddphi_j_lijk[indexes.matj == 1] = ddphi_1(quadrature_points)
    ddphi_j_lijk[indexes.matj == 2] = ddphi_2(quadrature_points)
    ddphi_j_lijk[indexes.matj == 3] = ddphi_3(quadrature_points)

    return ddphi_i_lijk, ddphi_j_lijk


# c)
def geth(parameters):
    # annahme abstände zwischen den Stützstellen sind konstant
    faktor = parameters.l / parameters.n
    h_1D = np.ones(parameters.n) * faktor
    # Erweiterung  in 2D
    h_2D = h_1D[:, np.newaxis]
    # Erweiterung  in 3D
    h_3D = h_2D[:, np.newaxis]

    return h_1D, h_2D, h_3D


# d)
def getTinv(parameters):
    quadrature_points = np.linspace(0, parameters.l, parameters.ns + 1)
    # Berechnet die Knotenpositionen
    x_l = np.arange(0, parameters.l, parameters.l / parameters.n)
    # Berechnet die Rücktransformation in 2D
    tinv_2D = np.outer(geth(parameters)[0], quadrature_points) + x_l[:, np.newaxis]
    # Erweiterung der Rücktransformation in 3D
    tinv_3D = tinv_2D[:, np.newaxis, :]
    # Erweiterung der Rücktransformation in 4D
    tinv_4D = tinv_2D[:, np.newaxis, np.newaxis, :]

    # Gibt die berechneten Rücktransformationen in 3D und 4D zurück
    return tinv_3D, tinv_4D


# e)
def getexp(parameters):
    i = np.arange(4)
    j = np.arange(4)
    J, I = np.meshgrid(j, i)
    delta_i_1 = (I == 1).astype(int)
    delta_i_3 = (I == 3).astype(int)
    delta_j_1 = (J == 1).astype(int)
    delta_j_3 = (J == 3).astype(int)

    exp_3d = np.stack([delta_i_1 + delta_i_3 + delta_j_1 + delta_j_3] * parameters.n, axis=0)

    exp_2d = np.stack([delta_i_1[:, 0] + delta_i_3[:, 0]] * parameters.n, axis=0)

    return exp_3d, exp_2d


'''
Aufgabe 19
'''


def getMbar_Aufgabe19(parameters, indexes, quadrature_points):
    h_3d = geth(parameters)[2]  # 0 = 1D, 1 = 2D, 2 = 3D
    exp = getexp(parameters)[0] + 1
    factors = np.power(h_3d, exp)

    phi_arrays = getphi(parameters, indexes, quadrature_points)
    vectorized_my = np.vectorize(parameters.my)
    integrand = vectorized_my(getTinv(parameters)[1]) * phi_arrays[0] * phi_arrays[1]
    integral = np.dot(integrand, getstencil(parameters))

    m_element = integral * factors

    return m_element


def getSbar_Aufgabe19(parameters, indexes, quadrature_points):
    h_3d = geth(parameters)[2]  # 0 = 1D, 1 = 2D, 2 = 3D
    exp = getexp(parameters)[0] - 3
    factors = np.power(h_3d, exp)

    ddphi_arrays = getddphi(parameters, indexes)
    vectorized_E = np.vectorize(parameters.E)
    vectorized_I = np.vectorize(parameters.E)
    t_inv = getTinv(parameters)[1]  # 4D array

    integrand = vectorized_E(t_inv) * vectorized_I(t_inv) * ddphi_arrays[0] * ddphi_arrays[1]
    integral = np.dot(integrand, getstencil(parameters))  # axis 3 means sum along k dimension

    s_element = integral * factors

    return s_element


def getqbar_Aufgabe19(parameters, indexes, quadrature_points):
    h_2D = geth(parameters)[1]
    exp = getexp(parameters)[1] + 1  # 0 = 3D, 1 = 2D
    factors = np.power(h_2D, exp)

    phi_arrays = getphi(parameters, indexes, quadrature_points)
    vectorized_q = np.vectorize(parameters.q)
    t_inv = getTinv(parameters)[0]

    integrand = vectorized_q(t_inv) * phi_arrays[2]
    integral = np.dot(integrand, getstencil(parameters))

    q_element = factors * integral

    return q_element


if __name__ == "__main__":

    time_start = time.time()

    n = 3  # Anzahl der Elemente in 1
    nh = 5  # Anzahl der zusätzlichen Auswertungspunkte je Element in 1
    ns = 7  # Ordnung der Quadratur in 1
    n_p = 100  # Anzahl der Zeitschritte in 1
    beta = 1 / 4  # Newmark- Koeffizient in 1
    gamma = 1 / 2  # Newmark- Koeffizient in 1
    eta = 0.1  # Zeitschrittweite in s
    l = 1  # Länge des Balkens in m
    x_0 = 1
    x = 1  # Zufälliges x für die lamda Funktionen für Aufgabe 1 - 15

    '''
    Aufgabe 15
    '''
    my = lambda x: x_0  # Längenspezifische Masse in kg/m
    E = lambda x: x_0  # Elastizitätsmodul in N/m^2
    I = lambda x: x_0  # Flächenträgheitsmoment in m^4
    q = lambda x: x_0  # Streckenlast in N/m

    '''
    B = np.array([[0, 1, 0],  # Auslenkung linkes Ende in m
                  [0, 2, 0],  # Anstieg linkes Ende in 1
                  [n, 3, 0],  # Moment rechtes Ende in Nm
                  [n, 4, 0]])  # Querkraft rechtes Ende in N
                      '''


    class Parameters:
        def __init__(self):
            self.n = n
            self.nh = nh
            self.ns = ns
            self.n_p = n_p
            self.beta = beta
            self.gamma = gamma
            self.eta = eta
            self.l = l
            self.x_0 = x_0
            self.my = my
            self.E = E
            self.I = I
            self.q = q
            self.h = l / n
            self.B = None

        def set_n(self, new_n):
            self.n = new_n
            self.B = np.array([[0, 1, 0],
                               [0, 2, 0],
                               [self.n, 3, 0],
                               [self.n, 4, 0]])
            self.h = self.l / self.n

        def set_q(self, new_q):
            self.q = lambda x: new_q


    class Indices:
        def __init__(self, index_matrices):
            self.matl = index_matrices[0]
            self.mati = index_matrices[1]
            self.matj = index_matrices[2]
            self.matlli = index_matrices[3]
            self.matllj = index_matrices[4]
            self.vekl = index_matrices[5]
            self.veki = index_matrices[6]
            self.veklli = index_matrices[7]


    params = Parameters()

    '''
    Aufgabe 10
    '''

    params.set_n(3)
    idx_10 = Indices(getindizes(params))
    alpha_e_static_solution_n3_arr = scipy.sparse.linalg.spsolve(getSe(params, idx_10),
                                                                 getve(params, idx_10))  # Find the static solution
    # alpha_e_static_solution_n3
    '''
    Aufgabe 11
    '''

    # Build the maticies for the DGL n = 100
    params.set_n(100)
    idx_11 = Indices(getindizes(params))  # update index matrices
    alpha_e_static_solution_n100 = scipy.sparse.linalg.spsolve(getSe(params, idx_11), getve(params, idx_11))

    # Plot both solutions
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 4), sharey='row')
    ax[0].plot(np.linspace(0, params.l, n + 1), alpha_e_static_solution_n3_arr[:2 * 3 + 2:2])
    ax[0].set_xlabel("x in m")
    ax[0].set_ylabel("w in m")
    ax[0].set_title("Solution for n = 3")
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(-0.15, 0.15)

    ax[1].plot(np.linspace(0, params.l, params.n + 1), alpha_e_static_solution_n100[:2 * params.n + 2:2])
    ax[1].set_title("Solution for n = 100")
    ax[1].set_ylabel("w in m")
    ax[1].set_xlabel("x in m")
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(-0.15, 0.15)
    plt.close()
   #  plt.show()

    '''
    Aufgabe 12
    '''

    params.set_n(3)
    params.set_q(0)
    idx_12 = Indices(getindizes(params))

    M_e = getMe(params, idx_12)
    M = getM(params, idx_12)
    S = getS(params, idx_12)
    S_e = getSe(params, idx_12)
    v_e = getve(params, idx_12)  # reinitialize the v_e vector, because the load has changed
    v_q = getvq(params, idx_12)
    v_n = getvn(params)
    C = getC(params)

    a_p_animation, total_energy_newmark = newmark_simmulation(params, alpha_e_static_solution_n3_arr)
    getplot(a_p_animation[:, : 2*params.n + 2:2], params)

    '''
    Aufgabe 13
    '''
    # set parameters
    params.set_n(3)
    params.set_q(1)
    idx_13 = Indices(getindizes(params))

    error_rates_plot = []
    n_plot = np.arange(1, 1001, 1)
    # iterate n from 1 to 1000
    for n_iteration in range(1, 1001, 1):
        # set parameters
        params.set_n(n_iteration)
        params.set_q(1)
        idx_13 = Indices(getindizes(params))

        x = np.linspace(0, params.l, n_iteration + 1)  # set supporting points

        # calculate analytic solution
        w_x = analytical_w(params, x)
        w_x_d = analytical_w_prime(params, x)
        # combine in one vektor
        w_ana = np.zeros(2 * len(w_x))
        w_ana[::2] = w_x
        w_ana[1::2] = w_x_d

        # calculate numeric solution
        alpha_static_solution = scipy.sparse.linalg.spsolve(getSe(params, idx_13), getve(params, idx_13))[:2 * len(w_x)]

        # create matrix A
        A = getA(params, idx_13)

        # calculate relative error
        numerator_error = (w_ana - alpha_static_solution).T @ A @ (w_ana - alpha_static_solution)
        denominator_error = w_ana.T @ A @ w_ana
        relative_error = np.sqrt(numerator_error) / np.sqrt(denominator_error)
        error_rates_plot.append(relative_error)

    n_error_test = 1000

    #     '''

    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Relative Error Rates over Varying number of Knots n")
    ax[0].plot(np.arange(n_error_test), error_rates_plot)
    ax[0].set_xlabel("n in 1")
    ax[0].set_ylabel("error_L^2 in 1")
    ax[1].plot(np.log(np.arange(n_error_test) + 1), np.log(error_rates_plot))
    ax[1].set_xlabel('log(n) in 1')
    ax[1].set_ylabel("log(error_L^2) in 1")
    plt.close()
    # plt.show()

    '''
    Aufgabe 14
    '''
    # calculate values
    params.set_n(3)
    t = np.linspace(0, 100, params.n_p)
    # a)
    total_energy_a = newmark_simmulation(params, alpha_e_static_solution_n3_arr)
    # b)
    params.eta = 1
    total_energy_b = newmark_simmulation(params, alpha_e_static_solution_n3_arr)
    # c)
    params.beta = 1
    params.gamma = 1
    params.eta = 0.1
    total_energy_c = newmark_simmulation(params, alpha_e_static_solution_n3_arr)
    # d)
    params.eta = 1
    total_energy_d = newmark_simmulation(params, alpha_e_static_solution_n3_arr)

    # create plot, 2x2-Grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Subplot (a)
    axs[0, 0].plot(t, total_energy_a[1], 'r')
    axs[0, 0].set_title('Gesamtenergie für eta = 0.1, beta = 0.25, gamma = 0.5')
    axs[0, 0].set_xlabel('t in s')
    axs[0, 0].set_ylabel('E_ges in J')
    axs[0, 0].set_xlim([0, 100])
    axs[0, 0].set_ylim([0, 0.05])

    # Subplot (b)
    axs[0, 1].plot(t, total_energy_b[1], 'g')
    axs[0, 1].set_title('Gesamtenergie für eta = 1, beta = 0.25, gamma = 0.5')
    axs[0, 1].set_xlabel('t in s')
    axs[0, 1].set_ylabel('E_ges in J')
    axs[0, 1].set_xlim([0, 100])
    axs[0, 1].set_ylim([0, 0.05])

    # Subplot (c)
    axs[1, 0].plot(t, total_energy_c[1], 'b')
    axs[1, 0].set_title('Gesamtenergie für eta = 0.1, beta = 1, gamma = 1')
    axs[1, 0].set_xlabel('t in s')
    axs[1, 0].set_ylabel('E_ges in J')

    # Subplot (d)
    axs[1, 1].plot(t, total_energy_d[1], 'k')
    axs[1, 1].set_title('Gesamtenergie für eta = 1, beta = 1, gamma = 1')
    axs[1, 1].set_xlabel('t in s')
    axs[1, 1].set_ylabel('E_ges in J')

    # Zeige das Ergebnis an
    plt.tight_layout()
    # plt.show()
    plt.close()

    '''
    Aufgabe 16
    '''
    # test für ns = 3
    params.set_n(3)
    params.ns = 3
    stencil = getstencil(params)
    #  print("stencil", stencil)

    '''
    Aufgabe 17, 18
    '''
    params.set_n(3)
    params.ns = 7
    idx_18 = Indices(getindizes(params))
    quadrature_points = np.linspace(0, params.l, params.ns + 1)
    a = getphi(params, idx_18, quadrature_points)
    # print("a", a)
    b = getddphi(params, idx_18)
    # print("b", b)
    h = geth(params)
    # print("h", h)
    T_inv = getTinv(params)
    # print("T_inv", T_inv)
    e3, e2 = getexp(params)
    # print("e3", e3, "e2", e2)

    '''
    Aufgabe 19
    '''

    idx_19 = Indices(getindizes(params))

    M_bar19 = getMbar_Aufgabe19(params, idx_19, quadrature_points)
    S_bar19 = getSbar_Aufgabe19(params, idx_19, quadrature_points)
    q_bar19 = getqbar_Aufgabe19(params, idx_19, quadrature_points)

    '''
    Aufgabe 20
    '''


    def getAl(parameters, indexes):
        # h values
        h_2D = geth(parameters)[1]
        exp = getexp(parameters)[1]  # 0 = 3D, 1 = 2D

        factors = np.power(h_2D, exp)[0]

        # equidistant points x_k
        k = np.arange(0, parameters.nh + 1)
        x_k = k / parameters.nh

        # get A-matrix
        A_elements = getphi(parameters, indexes, x_k)[3].T * factors
        A_elements = np.tile(A_elements, (parameters.n, 1, 1))

        # Generate indices
        J, K = np.meshgrid(np.arange(4), k)

        # Tile the index arrays to match the number of elements
        J_indices = np.tile(J, (parameters.n, 1, 1))
        K_indices = np.tile(K, (parameters.n, 1, 1))
        L = np.arange(parameters.n).reshape(parameters.n, 1, 1) * np.ones((1, parameters.nh + 1, 4)).astype(int)

        # get K_indices
        K_indices = (parameters.nh * L + K_indices).astype(int)

        # get J_indices
        J_indices = (2 * L + J_indices).astype(int)

        # Filter out duplicate assignments to avoid summing them in the COO matrix
        mask = np.zeros_like(A_elements.flatten(), dtype=bool)
        mask[np.unique(K_indices.flatten() * A_elements.shape[1] + J_indices.flatten(), return_index=True)[1]] = True

        # Assemble sparse matrix A
        A = coo_matrix((A_elements.flatten()[mask], (K_indices.flatten()[mask], J_indices.flatten()[mask]))).tocsr()

        return A


    Al = getAl(params, idx_19)

    test_solution = alpha_e_static_solution_n3_arr
    # !!!!!Later use a_p_animation!!!!!!!

    w = Al.dot(test_solution[:2 * params.n + 2])

    a_p_animation_high_res = np.array([Al.dot(solution[:2 * params.n + 2]) for solution in a_p_animation])

    getplot(a_p_animation_high_res, params)

    plt.show()
