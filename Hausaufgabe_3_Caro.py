# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 08:12:19 2024

@author: carol
"""

import numpy as np
import scipy.sparse
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

'''
Aufgabe 1, Initialisierung
'''

n = 3  # Anzahl der Elemente in 1
nh = 1  # Anzahl der zusätzlichen Auswertungspunkte je Element in 1
ns = 7  # Ordnung der Quadratur in 1
n_p = 100  # Anzahl der Zeitschritte in 1
beta = 1 / 4  # Newmark- Koeffizient in 1
gamma = 1 / 2  # Newmark- Koeffizient in 1
eta = 0.1  # Zeitschrittweite in s
l = 1  # Länge des Balkens in m
my = 1  # Längenspezifische Masse in kg/m
E = 1  # Elastizitätsmodul in N/m^2
I = 1  # Flächenträgheitsmoment in m^4
q = 1  # Streckenlast in N/m

B = np.array([[0, 1, 0],  # Auslenkung linkes Ende in m
              [0, 2, 0],  # Anstieg linkes Ende in 1
              [n, 3, 0],  # Moment rechtes Ende in Nm
              [n, 4, 0]])  # Querkraft rechtes Ende in N

'''
Aufgabe 2, nützliche Arrays
'''


# a) 3D-Arrays
def getindizes(n_elements: int):
    # a) 3D-Arrays
    nv = np.arange(0, 4, 1)  # Create an array with values [0, 1, 2, 3]
    J, I = np.meshgrid(nv, nv)  # repeats the rows, columns of nv in J, I

    matl = np.arange(n_elements).reshape(n_elements, 1, 1) * np.ones((1, 4, 4)).astype(
        int)  # Create a 3D array, where each element from 0 to n_elements-1 is repeated in shape(4,4)
    mati = np.repeat(I[np.newaxis, :, :], n_elements,
                     axis=0)  # Repeat the  array I along the first axis n_elements times
    matj = np.repeat(J[np.newaxis, :, :], n_elements,
                     axis=0)  # Repeat the  array J along the first axis n_elements times

    matlli = (2 * matl + mati).astype(int)  # Calculate a 3D array from  matl & mati
    matllj = (2 * matl + matj).astype(int)  # Calculate a 3D array from  matl & matj

    # We decided to use a row vector to represent the vector here
    # You can use a column vector as well [[[0], [0], [0], [0]], [[1], [1], ..
    # It does not matter that much as long as we stay consistent with this vector (matrix)
    # We use sparce matrices, so we flatten the vector anyway!
    veki, vekl = np.meshgrid(nv, np.arange(0,
                                           n_elements))  # Create 2D  arrays  using nv and an array from 0 to n_elements-1
    veklli = (2 * vekl + veki).astype(int)  # Calculate a 2D array from  vekl + veki

    # Print the generated 3D and 2D arrays
    print("3D-Array [l]", matl)
    print("3D-Array [i]", mati)
    print("3D-Array [j]", matj)
    print("3D-Array [2l + i]", matlli)
    print("3D-Array [2l + j]", matllj)
    print("2D-Array [l]", vekl)
    print("2D-Array [i]", veki)
    print("2D-Array [veklli]", veklli)

    # Return the generated arrays
    return matl, mati, matj, matlli, matllj, vekl, veki, veklli


'''
Aufgabe 3,4; Elementmatrizen, -vektoren
'''


# create Mass matrix
def getMbar(h, my, n_elements):
    faktor = my * h / 420  # define factor
    matrix = np.array(
        [[156, 22 * h, 54, -13 * h], [22 * h, 4 * h ** 2, 13 * h, -3 * h ** 2], [54, 13 * h, 156, -22 * h],
         [-13 * h, -3 * h ** 2, -22 * h, 4 * h ** 2]])  # define matrix
    M = faktor * matrix  # scale matrix by the factor
    M = np.tile(M, (n_elements, 1, 1))  # Replicate the scaled matrix for each element
    # Return the final 3D array
    return M


# create Stiffness matrix S the same way as M in getMbar
def getSbar(h, E, I, n_elements):
    faktor = E * I / h ** 3
    matrix = np.array([[12, 6 * h, -12, 6 * h], [6 * h, 4 * h ** 2, -6 * h, 2 * h ** 2], [-12, -6 * h, 12, -6 * h],
                       [6 * h, 2 * h ** 2, -6 * h, 4 * h ** 2]])
    S = faktor * matrix
    S = np.tile(S, (n_elements, 1, 1))
    return S


# create element vector vekq the same way as M in getMbar
def getqbar(h, q, n_elements):
    faktor = q * h / 12
    vektor = np.array([[6], [h], [6], [-h]])
    vekq = faktor * vektor
    vekq = np.tile(vekq, (n_elements, 1, 1))
    return vekq


'''
Aufgabe 5, Massen-, Steifigkeitsmatrix, Streckenlastvektor
'''
# Define indices
matl, mati, matj, matlli, matllj, vekl, veki, veklli = getindizes(n)


# Mass matrix
def getM(h, my, n_elements):
    M_alt = getMbar(h, my, n_elements)  # Define the data matrix
    M_neu = coo_matrix((M_alt.flatten(), (matlli.flatten(), matllj.flatten()))).tocsr()
    return M_neu


# Stiffness matrix
# Analogous to getM for the data of the stiffness matrix
def getS(h, E, I, n_elements):
    S_alt = getSbar(h, E, I, n_elements)
    S_neu = coo_matrix((S_alt.flatten(), (matlli.flatten(),
                                          matllj.flatten()))).tocsr()  # Convert the data matrix to a sparse matrix in COO format and then to CSR format
    return S_neu


# element vector
# Analogous to getM for the values of the element vector
def getvq(h, q, n_elements):
    vq_alt = getqbar(h, q, n_elements)
    vq_neu = coo_matrix((vq_alt.flatten(), (veklli.flatten(), np.zeros_like(veklli.flatten())))).tocsr()
    # np.zeros_like(veklli.flatten()) creates an array of zeros with the same shape as veklli.flatten()
    return vq_neu


'''
Aufgabe 6 
'''


def getC(n_elements):
    E1_indices = B[B[:, 1] == 1, 0]  # Extract indices for deflection
    E2_indices = B[B[:, 1] == 2, 0]  # Extract indices for slope

    # Combine indices for deflections and slope
    C1_indices = np.concatenate((E1_indices * 2, E2_indices * 2 + 1))

    num_entries = len(C1_indices)
    # Create sparse matrix C1 using the combined indices
    C1 = coo_matrix((np.ones(num_entries), (C1_indices, np.arange(num_entries))),
                    shape=(2 * n_elements + 2, num_entries)).tocsr()

    return C1


'''
Aufgabe 7
'''


def getvn(n_elements, B):
    E3_indices = B[B[:, 1] == 3, 0]  # Extract indices for moments
    E4_indices = B[B[:, 1] == 4, 0]  # Extract indices for shear forces

    c_3 = B[B[:, 1] == 3, 2]  # Extract values for moments
    c_4 = B[B[:, 1] == 4, 2]  # Extract values for shear forces

    # Create value arrays for moments & shear forces
    c_3_values = np.ones(len(E3_indices)) * c_3.T
    c_4_values = np.ones(len(E4_indices)) * c_4.T

    # Combine indices and values moments & shear forces
    v_N_rows = np.concatenate((E3_indices, E4_indices)).astype(int)
    v_N_cols = np.zeros(len(v_N_rows)).astype(int)
    v_N_vals = np.concatenate((c_3_values, c_4_values))
    v_N_shape = (2 * n_elements + 2, 1)

    # Create sparse matrix v_N with the combined indices and values
    v_N = coo_matrix((v_N_vals, (v_N_rows, v_N_cols)), v_N_shape).tocsr()

    return v_N


'''
Aufgabe 8
'''


def getvd(B):
    a_k_values = B[B[:, 1] == 1, 2]  # Extract deflection values and convert to column vector
    b_k_values = B[B[:, 1] == 2, 2]  # Extract slope values and convert to column vector
    # Create sparse matrices for c3 and c4
    a_k_sparse = scipy.sparse.csr_matrix(a_k_values)
    b_k_sparse = scipy.sparse.csr_matrix(b_k_values)
    # Combine the vectors
    vD = scipy.sparse.vstack([a_k_sparse, b_k_sparse])

    return vD


'''
Aufgabe 9
'''


# ist zero filler richtig definiert?


# a Mass matrix
def getMe(h, my, n):
    M = getM(h, my, n)
    C = getC(n)
    C0 = scipy.sparse.csr_matrix(np.zeros_like(C.toarray()))
    zero_filler = scipy.sparse.csr_matrix(np.zeros((C.shape[1], len(C0.toarray()[0]))))
    # horizontally merge M and C0
    M_C0_horizontal_stack = scipy.sparse.hstack([M, C0])
    # horizontally merge C0 and zero filler
    filler_C0_horizontal_stack = scipy.sparse.hstack([C0.T, zero_filler])
    # vertically merge M_C0 and filler_C0
    Me = scipy.sparse.vstack([M_C0_horizontal_stack, filler_C0_horizontal_stack])

    return Me


# b Stiffness matrix
def getSe(h, E, I, n):
    S = getS(h, E, I, n)
    C = getC(n)
    CT = C.T
    zero_filler = scipy.sparse.csr_matrix(np.zeros((C.shape[1], len(C.toarray()[0]))))
    # horizontally merge S and C
    S_C_horizontal_stack = scipy.sparse.hstack([S, C])
    # horizontally merge CT and zero filler
    filler_CT_horizontal_stack = scipy.sparse.hstack([CT, zero_filler])
    # vertically merge S_C and filler_CT
    Se = scipy.sparse.vstack([S_C_horizontal_stack, filler_CT_horizontal_stack])

    return Se


# c Element vector
def getve(h, q, n, B):
    vq = getvq(h, q, n)
    vN = getvn(n, B)
    # create 1st row
    ve1 = vq + vN
    # create 2nd row
    vD = getvd(B)
    # merge and stack vertically
    ve = scipy.sparse.vstack([ve1, vD])

    return ve


'''
Aufgabe 10, Statik, Beschleunigung = 0
'''
# define h as segment length
h = l / n
# stiffness matrix
SE = getSe(h, E, I, n).tocsr()
# element vector
vE = getve(h, q, n, B)
# solve the system of equations
alpha_e_static_solution_n = scipy.sparse.linalg.spsolve(SE, vE)
alpha_e_static_solution_n3 = scipy.sparse.csr_matrix(alpha_e_static_solution_n)

'''
Aufgabe 11, pLot
'''


# n = 3
def getplot(alpha_e_static_solution_n3, l, n):
    # Calculate support points
    x = np.linspace(0, l, n + 1)
    # Slice values to positions xi from the static solution
    y = alpha_e_static_solution_n3[:2 * n + 2:2]
    # Generate plot
    plt.figure(figsize=(5, 5))
    plt.plot(x, y)
    plt.title(f"Bending line for n = {n}")
    plt.xlabel("x in m")
    plt.ylabel("w in m")
    plt.ylim(-max(y)*1.2, max(y)*1.2)
    plt.xlim(0, l * 1.2)
    plt.show()


getplot(alpha_e_static_solution_n, l, n)



