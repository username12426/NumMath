import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
factorial = np.vectorize(np.math.factorial)

# Programmieraufgabe 1

n = 3   # Anzahl der Elemente in
n_hat = 1   # Anzahl der zusätzlichen Auswertungspunkte je Element in 1
n_und_tilde = 7  # Ordnung der Quadratur in 1, _und denotes an underline
n_p = 100   # Anzahl der Zeitschritte in 1
beta = 1/4  # Newmark-Koeffizient in 1
gamma = 1/2     # Newmark-Koeffizient in 1
eta = 0.1   # Zeitschrittweite in s
l = 1   # Länge des Balkens in m
my = 1  # Längenspezifische Masse in kg/m
E = 1   # Elastizitätsmodul in Newton/m^2
I = 1   # Elastizitätsmodul in m^4
q = 1   # Streckenlast in Newton/m

B = np.array([[0, 1, 0],    # Auslenkung linkes Ende in m
              [0, 2, 0],    # Anstieg linkes Ende in 1
              [n, 3, 0],    # Moment rechtes Ende in Nm
              [n, 4, 0]])   # Querkraft rechtes Ende in N

# Prorammiraufgabe 2 a


def getindizes():
    l_L = np.array(range(0, n))  # range does the -1 by default
    i_und = np.array([0, 1, 2, 3])
    j_und = np.array([0, 1, 2, 3])

    J, I = np.meshgrid(j_und, i_und)    # Create meshgrid, because they have the desired appearance

    L_matrix_a = np.zeros((len(l_L), len(i_und), len(j_und)))     # Crete arrays with shape(n, 4, 4)
    I_matrix_a = np.zeros((len(l_L), len(i_und), len(j_und)))
    J_matrix_a = np.zeros((len(l_L), len(i_und), len(j_und)))

    for i in l_L:
        L_matrix_a[i, :, :] = i   #
        I_matrix_a[i] = I   # Meshgrid of I stacked n times
        J_matrix_a[i] = J   # Meshgrid of J stacked n times

    two_l_plus_i_a = 2 * L_matrix_a + I_matrix_a
    two_l_plus_j_a = 2 * L_matrix_a + J_matrix_a

    # Programmieraufgabe 2 b

    L_matrix_b = np.zeros((len(l_L), len(i_und), 1))
    I_matrix_b = np.zeros((len(l_L), len(i_und), 1))

    J, I = np.meshgrid(j_und, 1)
    J_T = J.T

    for i in l_L:
        L_matrix_b[i, :, 0] = i
        I_matrix_b[i, :] = J_T

    two_l_plus_i_b = 2 * L_matrix_b + I_matrix_b

    return L_matrix_a, I_matrix_a, J_matrix_a, two_l_plus_i_a, two_l_plus_j_a, two_l_plus_i_b

getindices()

# Programmieraufgabe 3

def getMbar(h):
    Mbar_matrix = np.array([[156, 22 * h, 54, -13 * h],
                            [22 * h, 4 * h ** 2, 13 * h, -3 * h ** 2],
                            [54, 13 * h, 156, -22],
                            [-13 * h, -3 * h ** 2, -22 * h, 4 * h ** 2]])

    return ((my * h) / 420) * Mbar_matrix


def getSbar(h):
    Sbar_matrix = np.arry([[12, 6*h, -12, 6*h],
                           [6*h, 4*h**2, -6*h, 2*h**2],
                           [-12, -6*h, 12, -6*h],
                           [6*h, 2*h**2, -6*h, 4*h**2]])

    return ((E*I)/h**3) * Sbar_matrix


def getqbar(h):
    return ((q * h)/12) * np.array([[6], [h], [6], [-h]])


# Programmieraufgabe 4

def getMbar(h) -> np.ndarray:
    Mbar_matrix = np.array([[156, 22 * h, 54, -13 * h],
                            [22 * h, 4 * h ** 2, 13 * h, -3 * h ** 2],
                            [54, 13 * h, 156, -22],
                            [-13 * h, -3 * h ** 2, -22 * h, 4 * h ** 2]])

    Mbar_matrix = ((my * h) / 420) * Mbar_matrix
    stacked_Mbar_matrix = np.zeros((n, 4, 4))
    for i in range(n):
        stacked_Mbar_matrix[i] = Mbar_matrix

    return stacked_Mbar_matrix


def getSbar(h):
    Sbar_matrix = np.arry([[12, 6 * h, -12, 6 * h],
                           [6 * h, 4 * h ** 2, -6 * h, 2 * h ** 2],
                           [-12, -6 * h, 12, -6 * h],
                           [6 * h, 2 * h ** 2, -6 * h, 4 * h ** 2]])

    Sbar_matrix = ((E * I) / h ** 3) * Sbar_matrix
    stacked_Sbar_matrix = np.arry((n, 4, 4))
    for i in range(n):
        stacked_Sbar_matrix[i] = Sbar_matrix

    return stacked_Sbar_matrix


def getqbar(h):
    qbar_vector = ((q * h)/12) * np.array([[6], [h], [6], [-h]])
    stacked_qbar_matrix = np.zeros((n, 4, 1))
    print(stacked_qbar_matrix)
    for i in range(n):
        stacked_qbar_matrix[i] = qbar_vector

    return stacked_qbar_matrix


# Programmieraufgabe 5

_, _, _, index_matrix_i, index_matrix_j, index_vector_i = getindizes()

# For getM and getS you need to use the index Matrix because the output is a matrix
# For getvq you use the index vector because the output is a vector

def getM(h):
    M_element = getMbar(h)
    M_assemble = coo_matrix((M_element.flatten(), (index_matrix_i.flatten(), index_matrix_j.flatten()))).tocsr()
    # You can use the flatten function to avoid a lot of indexing
    return M_assemble


def getS(h):
    S_element = getSbar(h)
    S_assemble = coo_matrix((S_element.flatten(), (index_matrix_i.flatten(), index_matrix_j.flatten()))).tocsr()
    return S_assemble


def getvq(h):
    vq_element = getqbar(h)
    vq_assemble = coo_matrix((vq_element.flatten(), (index_vector_i.flatten(), np.zeros_like(index_vector_i.flatten())))).tocsr()
    return vq_assemble






















