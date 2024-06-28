import numpy as np
import scipy.sparse
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
import matplotlib.animation as animation

'''
Aufgabe 1, Initialisierung
'''

n = 67  # Anzahl der Elemente in 1
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


# We need this function, because when we change the n, the B matrix is changes as well
def reinitialize_B(n_element):
    global B
    B = np.array([[0, 1, 0],  # Auslenkung linkes Ende in m
                  [0, 2, 0],  # Anstieg linkes Ende in 1
                  [n_element, 3, 0],  # Moment rechtes Ende in Nm
                  [n_element, 4, 0]])  # Querkraft rechtes Ende in N


'''
Aufgabe 2, nützliche Arrays
'''


# a) 3D-Arrays
def getindizes(n_elements: int):
    # a) 3D-Arrays
    nv = np.arange(0, 4, 1)  # Hilfsvektor
    J, I = np.meshgrid(nv, nv)  #

    matl = np.arange(n_elements).reshape(n_elements, 1, 1) * np.ones((1, 4, 4)).astype(int)
    mati = np.repeat(I[np.newaxis, :, :], n_elements, axis=0)
    matj = np.repeat(J[np.newaxis, :, :], n_elements, axis=0)

    matlli = (2 * matl + mati).astype(int)
    matllj = (2 * matl + matj).astype(int)

    '''
    We decided to use a row vector to represent the vectors here, because it is simpler to create
    You can use a column vector as well [[[0], [0], [0], [0]], [[1], [1], ..
    It does not matter that much as long as we stay consistent with this vector (matrix)
    We use sparce matrices, so we flatten the vector anyway!
    '''

    veki, vekl = np.meshgrid(nv, np.arange(0, n_elements))
    veklli = (2 * vekl + veki).astype(int)

    return matl, mati, matj, matlli, matllj, vekl, veki, veklli


'''
Aufgabe 3,4; Elementmatrizen, -vektoren
'''

# We are still not sure if we should pass all variables as parameters in the function
# currently "my", "E" and "I" are defined outside the functions. We will change this if it leads
# to problems

def getMbar(h, n_elements):
    faktor = my * h / 420
    matrix = np.array(
        [[156, 22 * h, 54, -13 * h], [22 * h, 4 * h ** 2, 13 * h, -3 * h ** 2], [54, 13 * h, 156, -22 * h],
         [-13 * h, -3 * h ** 2, -22 * h, 4 * h ** 2]])
    M = faktor * matrix
    M = np.tile(M, (n_elements, 1, 1))
    return M



def getSbar(h, n_elements):
    faktor = E * I / h ** 3
    matrix = np.array([[12, 6 * h, -12, 6 * h], [6 * h, 4 * h ** 2, -6 * h, 2 * h ** 2], [-12, -6 * h, 12, -6 * h],
                       [6 * h, 2 * h ** 2, -6 * h, 4 * h ** 2]])
    S = faktor * matrix
    S = np.tile(S, (n_elements, 1, 1))
    return S


def getqbar(h, n_elements):
    faktor = q * h / 12
    vektor = np.array([[6], [h], [6], [-h]])
    vekq = faktor * vektor
    vekq = np.tile(vekq, (n_elements, 1, 1))
    return vekq


'''
Aufgabe 5, Massen-, Steifigkeitsmatrix, Streckenlastvektor
'''

'''
We are not sure if we should pass the matlli, matllj.. into the function as a Parameter
Both versions would work
If we pass the matricies as a parameter we have a lot of parameters
If we dont pass them, 
'''

# Massenmatrix
def getM(h: float, n_elements: int):
    M_alt = getMbar(h, n_elements)  # daten Matrix definieren
    M_neu = coo_matrix((M_alt.flatten(), (matlli.flatten(), matllj.flatten()))).tocsr()
    return M_neu


# Steifigkeitsmatrix
# analog zu getM für die Daten der Steifigkeitsmatrix
def getS(h, n_elements):
    S_alt = getSbar(h, n_elements)
    print(S_alt.shape)
    print(matlli.shape)
    print(matllj.shape)
    S_neu = coo_matrix((S_alt.flatten(), (matlli.flatten(), matllj.flatten()))).tocsr()
    return S_neu


# Streckenlastvektor
# analog zu getM für werte des Streckenlastvektors
def getvq(h, n_elements):
    vq_alt = getqbar(h, n_elements)
    vq_neu = coo_matrix((vq_alt.flatten(), (veklli.flatten(), np.zeros_like(veklli.flatten())))).tocsr()
    return vq_neu


'''
Aufgabe 6 Variante 1
'''

# Sind und nicht ganz sicher ob wir "==" verwenden dürfen, deshalb gubt es zwei Varianten
# B is not getting passed into the function, so we have to be careful when we redefine the B matrix!!

def getC(n_elements):

    E1_indices = B[B[:, 1] == 1, 0]
    E2_indices = B[B[:, 1] == 2, 0]

    C1_indices = np.concatenate((E1_indices * 2, E2_indices * 2 + 1))

    assert max(C1_indices) <= n_elements, "Update the B matrix for current n"   # This does not catch all errors!

    num_entries = len(C1_indices)
    C1 = coo_matrix((np.ones(num_entries), (C1_indices, np.arange(num_entries))), shape=(2 * n_elements + 2, num_entries)).tocsr()

    return C1


# Version 2
# Here you only need to know ho many 1 and 2 conditions there are


'''
def getC(n_elements):
    # because we are not allowed to use this, because of the "==" operator, you can use this
    B_sorted = np.zeros((4, n_elements + 1))
    B_sorted[B[:, 1] - 1, B[:, 0]] = 1
    E1_count = np.sum(B_sorted[0], dtype=int)
    E2_count = np.sum(B_sorted[1], dtype=int)

    # Build the index vectors
    E1_values = np.ones(E1_count)  # all values are ones
    E1_rows = np.arange(0, E1_count)  # number of constrains
    E1_cols = B[:E1_count, 0] * 2  # * 2 from j = 2k (formula)
    E1_shape = (2 * n_elements + 2, E1_count)  # 2n+2 is number of constraints for all knots

    E2_values = np.ones(E2_count)
    E2_rows = np.arange(0, E2_count)
    E2_cols = B[:E2_count, 0] * 2 + 1  # * 2 + 1 from j = 2k+1
    E2_shape = (2 * n_elements + 2, E2_count)

    E1 = coo_matrix((E1_values, (E1_cols, E1_rows)), shape=E1_shape).tocsr()
    E2 = coo_matrix((E2_values, (E2_cols, E2_rows)), shape=E2_shape).tocsr()

    return scipy.sparse.hstack([E1, E2])
    ...
    ...
    
'''


# Aufgabe 7

# Variante 1. Das ist wieder die Variante mit dem ==, man kann das aber genau so auch alternativ ohne machen
# The B matrix has to be updated to the current n before you use this function

def getvn(n_elements):

    E3_indices = B[B[:, 1] == 3, 0]     # get all elements for K3, and extract the indices
    E4_indices = B[B[:, 1] == 4, 0]

    c_3 = B[B[:, 1] == 3, 2]    # get all elementd for K3 and extract the values to the indices
    c_4 = B[B[:, 1] == 4, 2]

    c_3_values = np.ones(len(E3_indices)) * c_3.T
    c_4_values = np.ones(len(E4_indices)) * c_4.T

    v_N_rows = np.concatenate((E3_indices, E4_indices)).astype(int)

    assert max(v_N_rows) <= n_elements, "Update the B matrix for current n"  # check for update errors

    v_N_cols = np.zeros(len(v_N_rows)).astype(int)
    v_N_vals = np.concatenate((c_3_values, c_4_values))
    v_N_shape = (2 * n_elements + 2, 1)

    v_N = coo_matrix((v_N_vals, (v_N_rows, v_N_cols)), v_N_shape).tocsr()

    return v_N


# Aufgabe 8

def getvd():
    a_k_values = B[B[:, 1] == 1, 2]
    b_k_values = B[B[:, 1] == 2, 2]

    v_D_values = np.concatenate((a_k_values, b_k_values)).astype(int)
    v_D_rows = np.arange(len(v_D_values)).astype(int)
    v_D_cols = np.zeros(len(v_D_values)).astype(int)

    v_D = coo_matrix((v_D_values, (v_D_rows, v_D_cols))).tocsr()

    return v_D

# Aufgabe 9

def getMe(h, n_elements):
    M = getM(h, n_elements)
    C = getC(n_elements)
    C0 = np.zeros_like(C.toarray())
    I, J = np.meshgrid(np.arange(2), np.arange(2))
    zero_filler = coo_matrix((np.zeros(4), (I.flatten(), J.flatten()))).tocsr()

    M_C0_horizontal_stack = scipy.sparse.hstack([M, C0])
    filler_C0_horizontal_stack = scipy.sparse.hstack([C0.T, zero_filler])

    Me = scipy.sparse.vstack([M_C0_horizontal_stack, filler_C0_horizontal_stack])
    return Me


def getSe(h, n_elements):
    S = getS(h, n_elements)
    C = getC(n_elements)
    I, J = np.meshgrid(np.arange(2), np.arange(2))
    zero_filler = coo_matrix((np.zeros(4), (I.flatten(), J.flatten()))).tocsr()

    S_C_horizontal_stack = scipy.sparse.hstack([S, C])
    C_filler_horizontal_stack = scipy.sparse.hstack([C.T, zero_filler])
    Se = scipy.sparse.vstack([S_C_horizontal_stack, C_filler_horizontal_stack]).tocsr()
    return Se


def getve(h, n_elements):
    vq_nq = getvq(h, n_elements) + getvn(n_elements)
    v_E = scipy.sparse.vstack([vq_nq, getvd()])
    return v_E

'''
Aufgabe 10
'''

n = 3
h = l / n

reinitialize_B(n)   # We changed the n to we have to update the B matrix
matl, mati, matj, matlli, matllj, vekl, veki, veklli = getindizes(n)    # Build the matricies for the DGL n = 3
alpha_e_static_solution_n3 = scipy.sparse.linalg.spsolve(getSe(h, n), getve(h, n))  # Find the static solution

'''
Aufgabe 11
'''

# Build the maticies for the DGL n = 100
n_100 = 100
h = l / n_100
reinitialize_B(n_100)   # update the B matrix for the new n
matl, mati, matj, matlli, matllj, vekl, veki, veklli = getindizes(n_100)    # update index matrices
alpha_e_static_solution_n100 = scipy.sparse.linalg.spsolve(getSe(h, n_100), getve(h, n_100))

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 4), sharey='row')
ax[0].plot(np.arange(0, l, l/(n+1)), alpha_e_static_solution_n3[:2*n+2:2])
ax[0].set_xlabel("x in m")
ax[0].set_ylabel("w in m")
ax[0].set_title(f"Solution for n = {n}")
ax[1].plot(np.arange(0, l, l/(n_100 + 1)), alpha_e_static_solution_n100[:2*n_100+2:2])
ax[1].set_title("Solution for n = 100")
ax[1].set_ylabel("w in m")
ax[1].set_xlabel("x in m")

plt.show()


'''
Aufgabe 12
'''

n = 3
q = 0   # Keine Streckenlast
h = l/n
reinitialize_B(n)   # update the B matrix for the new n
matl, mati, matj, matlli, matllj, vekl, veki, veklli = getindizes(n)

M = getMe(h, n)
S = getSe(h, n)
v_e = getve(h, n)   # reinitialize the v_e vector, because the load has changed

# use the static state as our startingpoint for the Newmark-algorithim

a_0_rows = np.arange(len(alpha_e_static_solution_n3), dtype=int)
a_0_cols = np.zeros_like(alpha_e_static_solution_n3, dtype=int)

a_p = coo_matrix((alpha_e_static_solution_n3, (a_0_rows, a_0_cols))).tocsr()    # deflections
a_d_p = coo_matrix((np.zeros_like(alpha_e_static_solution_n3), (a_0_rows, a_0_cols))).tocsr()   # velocities

# In the static case there is no acceleration
a_dd_p = coo_matrix((np.zeros_like(alpha_e_static_solution_n3), (a_0_rows, a_0_cols))).tocsr()  # acceleration


a_p_animation = np.zeros((100, 2*n+2))  # Data Matrix for the Animation

# Iterate over n_p timesteps using the Newmark-Algorithm
for time_step in range(n_p):
    a_explicit = a_p + (a_d_p*my) + (0.5 - beta)*(a_dd_p*my**2)
    a_d_explicit = a_d_p + (1 - gamma)*(a_dd_p*my)

    a_dd_p = scipy.sparse.linalg.spsolve(M + (S*beta*my**2), v_e - S.dot(a_explicit))
    a_dd_p = coo_matrix((a_dd_p, (a_0_rows, a_0_cols))).tocsr()

    a_p = a_explicit + beta*a_dd_p*my**2

    a_p_animation[time_step,:] = a_p.toarray()[:2*n+2].T    # only the first 2n+2 elements are coordinates

    a_d_p = a_d_explicit + gamma*a_dd_p*my


def update_frame(frame):
    plt.cla()  # Clear current plot

    x_knots = np.arange(n + 1) / (n + 1)  # we have n+1 deflections (for each knot)

    plt.plot(x_knots, a_p_animation[frame, :2 * n + 2:2])  # Only pot the deviations, not the forces of the alpha vector

    plt.xlim(0, 1)
    plt.ylim(-0.5, 0.5)
    plt.title(f'Swinging Beam Animation for n = {n}')
    plt.xlabel('x in m')
    plt.ylabel('w in m')


def getplot():
    frames = a_p_animation.shape[0]  # set up animation

    fig, ax = plt.subplots()
    _ = animation.FuncAnimation(fig, update_frame, frames=frames, interval=100, repeat=False)
    plt.show()
