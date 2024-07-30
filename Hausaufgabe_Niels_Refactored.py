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

'''
Aufgabe 2, nützliche Arrays
'''

# a) 3D-Arrays
def getindizes(parameters:object):
    # a) 3D-Arrays
    nv = np.arange(0, 4, 1)  # Create an array with values [0, 1, 2, 3]
    J, I = np.meshgrid(nv, nv)  #repeats the rows, columns of nv in J, I

    matl = np.arange(parameters.n).reshape(parameters.n, 1, 1) * np.ones((1, 4, 4)).astype(int) #Create a 3D array, where each element from 0 to parameters.n-1 is repeated in shape(4,4)
    mati = np.repeat(I[np.newaxis, :, :], parameters.n, axis=0) #Repeat the  array I along the first axis parameters.n times
    matj = np.repeat(J[np.newaxis, :, :], parameters.n, axis=0) #Repeat the  array J along the first axis parameters.n times

    matlli = (2 * matl + mati).astype(int) #Calculate a 3D array from  matl & mati
    matllj = (2 * matl + matj).astype(int) #Calculate a 3D array from  matl & matj

    # We decided to use a row vector to represent the vector here
    # You can use a column vector as well [[[0], [0], [0], [0]], [[1], [1], ..
    # It does not matter that much as long as we stay consistent with this vector (matrix)
    # We use sparce matrices, so we flatten the vector anyway!
    veki, vekl = np.meshgrid(nv, np.arange(0, parameters.n)) #Create 2D  arrays  using nv and an array from 0 to parameters.n-1
    veklli = (2 * vekl + veki).astype(int) #Calculate a 2D array from  vekl + veki

    # Return the generated arrays
    return matl, mati, matj, matlli, matllj, vekl, veki, veklli


'''
Aufgabe 3,4; Elementmatrizen, -vektoren
'''

#create Mass matrix
def getMbar(parameters:object):
    h = parameters.h
    faktor = parameters.my(x) * h / 420 # define factor
    matrix = np.array(
        [[156, 22 * h, 54, -13 * h], [22 * h, 4 * h ** 2, 13 * h, -3 * h ** 2], [54, 13 * h, 156, -22 * h],
         [-13 * h, -3 * h ** 2, -22 * h, 4 * h ** 2]]) # define matrix
    M = faktor * matrix # scale matrix by the factor
    M = np.tile(M, (parameters.n, 1, 1)) # Replicate the scaled matrix for each element
    # Return the final 3D array
    return M


# create Stiffness matrix S the same way as M in getMbar
def getSbar(parameters:object):
    h = parameters.h
    faktor = parameters.E(x) * parameters.I(x) / h ** 3
    matrix = np.array([[12, 6 * h, -12, 6 * h], [6 * h, 4 * h ** 2, -6 * h, 2 * h ** 2], [-12, -6 * h, 12, -6 * h],
                       [6 * h, 2 * h ** 2, -6 * h, 4 * h ** 2]])
    S = faktor * matrix
    S = np.tile(S, (parameters.n, 1, 1))
    return S

# create element vector vekq the same way as M in getMbar
def getqbar(parameters:object):
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
def getM(parameters:object, indexes:object):
    M_alt = getMbar(parameters)  # Define the data matrix
    M_neu = coo_matrix((M_alt.flatten(), (indexes.matlli.flatten(), indexes.matllj.flatten()))).tocsr()
    return M_neu


# Stiffness matrix
# Analogous to getM for the data of the stiffness matrix
def getS(parameters, indexes):
    S_alt = getSbar(parameters)
    S_neu = coo_matrix((S_alt.flatten(), (indexes.matlli.flatten(), indexes.matllj.flatten()))).tocsr()# Convert the data matrix to a sparse matrix in COO format and then to CSR format
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

    assert max(C1_indices) <= parameters.n, "Update the B matrix for current n"   # This does not catch all errors!

    num_entries = len(C1_indices)
    C1 = coo_matrix((np.ones(num_entries), (C1_indices, np.arange(num_entries))), shape=(2 * parameters.n + 2, num_entries)).tocsr()

    return C1


'''
Aufgabe 7
'''

# Variante 1. Das ist wieder die Variante mit dem ==, man kann das aber genau so auch alternativ ohne machen
# The B matrix has to be updated to the current n before you use this function

def getvn(parameters):

    E3_indices = parameters.B[parameters.B[:, 1] == 3, 0]     # get all elements for K3, and extract the indices
    E4_indices = parameters.B[parameters.B[:, 1] == 4, 0]

    c_3 = parameters.B[parameters.B[:, 1] == 3, 2]    # get all elementd for K3 and extract the values to the indices
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
Aufgabe 12
'''


def newmark_simmulation(parameters, static_solution):

    beta = parameters.beta
    gamma = parameters.gamma
    eta = parameters.eta

    # use the static state as our startingpoint for the Newmark-algorithim

    a_0_rows = np.arange(len(static_solution), dtype=int)
    a_0_cols = np.zeros_like(static_solution, dtype=int)

    a_p = coo_matrix((static_solution, (a_0_rows, a_0_cols))).tocsr()    # deflections

    # In the static case there is no acceleration and initial velocity
    a_d_p = coo_matrix((np.zeros_like(static_solution), (a_0_rows, a_0_cols))).tocsr()   # velocities = 0
    a_dd_p = coo_matrix((np.zeros_like(static_solution), (a_0_rows, a_0_cols))).tocsr()  # acceleration = 0


    a_p_animation = np.zeros((parameters.n_p, 2*parameters.n+2))  # Data Matrix for the Animation
    total_energy_timesteps = np.zeros(parameters.n_p)  # for Task 14

    # Iterate over n_p timesteps using the Newmark-Algorithm
    for time_step in range(parameters.n_p):
        a_explicit = a_p + (a_d_p*eta) + (0.5 - beta)*(a_dd_p*eta**2)
        a_d_explicit = a_d_p + (1 - gamma)*(a_dd_p*eta)

        a_dd_p = scipy.sparse.linalg.spsolve(M_e + (S_e*beta*eta**2), v_e - S_e.dot(a_explicit))
        a_dd_p = coo_matrix((a_dd_p, (a_0_rows, a_0_cols))).tocsr()

        a_p = a_explicit + beta*a_dd_p*eta**2

        a_p_animation[time_step,:] = a_p.toarray()[:2*parameters.n+2].T    # only the first 2n+2 elements are coordinates

        a_d_p = a_d_explicit + gamma*a_dd_p*eta

        loads_v = a_p[2*parameters.n+2:]

        total_energy = 0.5*a_d_p[:2*parameters.n+2].T @ M @ a_d_p[:2*parameters.n+2] + (0.5*S*a_p[:2*parameters.n+2] - v_q - C @ loads_v - v_n).T @ a_p[:2*parameters.n+2]
        total_energy_timesteps[time_step] = total_energy[0].toarray()

    return a_p_animation, total_energy_timesteps


def update_frame(frame):
    plt.cla()  # Clear current plot

    x_knots = np.arange(n + 1) / (n + 1)  # we have n+1 deflections (for each knot)
    plt.plot(x_knots, a_p_animation[frame, :2 * n + 2:2])  # Only polt the deviations, not the forces of the alpha vector

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


'''
Aufgabe 13
'''

# This task is about the static case so the load is not zero q != 0


def analytical_w(parameters, x_k):
    return (parameters.q(x)/(parameters.E(x)*parameters.I(x))) * ((x_k**4)/24 - (parameters.l*x_k**3)/6 + ((parameters.l**2)*(x_k**2))/4)


def analytical_w_prime(parameters, x_k):
    return (parameters.q(x)/(parameters.E(x)*parameters.I(x))) * ((x_k**3)/6 - (parameters.l*x_k**2)/2 + (parameters.l**2*x_k)/2)


'''
Aufgabe 14
'''

'''
Aufgabe 15
'''

'''
Aufgabe 16
'''


def getstencil(quadrature_points):

    # Berechnung des stencils auf einem Referenzelement

    I, K = np.meshgrid(np.arange(len(quadrature_points)), quadrature_points)
    vandermonde_matrix = np.power(K, I)
    v_plus1 = 1 / (I[0] + 1)
    # We can interpret the formula for a as the solution of a system of equations
    s = np.linalg.solve(vandermonde_matrix.T, v_plus1)

    return s


'''
Aufgabe 17
'''

'''
def getphi(quadrature_points):

    # calculate the base function values on a reference element
    # I don't understand why n is given in the task

    evaluated_points = np.array([[1 - 3*(quadrature_points**2) + 2*(quadrature_points**3)],
                                 [quadrature_points - 2*(quadrature_points**2) + quadrature_points**3],
                                 [3*(quadrature_points**2) - 2*(quadrature_points**3)],
                                 [-(quadrature_points**2) + (quadrature_points**3)]])

    return evaluated_points

'''

'''
def getddphi(quadrature_points):
    evaluated_points = np.array([[-6 + 12*quadrature_points],
                                 [-4 + 6*quadrature_points],
                                 [6 - 12*quadrature_points],
                                 [-2 + 6*quadrature_points]])

    return evaluated_points
'''

'''
def geth(parameters):
    # This only works under the assumption that all beam elements are equally spaced
    # Wich they have to be because the mass matrix was derived using this assumption
    return np.ones(parameters.n) * parameters.l/parameters.n
    
'''

'''
def getTinv(parameters, quadrature_points):
    # assuming an equidistant beam elements
    # x_l are the first n-1 knot positions (base points of the reference coordinates)
    x_l = np.arange(0, parameters.l, parameters.l/parameters.n)
    return np.outer(geth(parameters.n, parameters.l), quadrature_points) + x_l[:, np.newaxis]
'''

def getexp(parameters):
    i = np.arange(parameters.n+1)
    j = np.arange(parameters.n+1)
    J, I = np.meshgrid(j, i)
    delta_i_1 = (I == 1).astype(int)
    delta_i_3 = (I == 3).astype(int)
    delta_j_1 = (J == 1).astype(int)
    delta_j_3 = (J == 3).astype(int)

    exp_3d = np.stack([delta_i_1 + delta_i_3 + delta_j_1 + delta_j_3] * parameters.n, axis=0)

    exp_2d = np.stack([delta_i_1[:, 0] + delta_i_3[:, 0]] * parameters.n, axis=0)

    return exp_3d, exp_2d


'''
Aufgabe 18
'''


def getphi(parameters:object, indexes:object, quadrature_points):
    phi = np.array([1 - 3 * (quadrature_points ** 2) + 2 * (quadrature_points ** 3),
                                 quadrature_points - 2 * (quadrature_points ** 2) + quadrature_points ** 3,
                                 3 * (quadrature_points ** 2) - 2 * (quadrature_points ** 3),
                                 -(quadrature_points ** 2) + (quadrature_points ** 3)])

    phi_i_lijk = np.zeros((parameters.n, 4, 4, len(quadrature_points)))
    phi_j_lijk = np.zeros((parameters.n, 4, 4, len(quadrature_points)))
    phi_i_lik = np.zeros((parameters.n, 4, len(quadrature_points)))

    phi_i_lik[indexes.veki == 0] = phi[0]
    phi_i_lik[indexes.veki == 1] = phi[1]
    phi_i_lik[indexes.veki == 2] = phi[2]
    phi_i_lik[indexes.veki == 3] = phi[3]

    phi_i_lijk[indexes.mati == 0] = phi[0]
    phi_i_lijk[indexes.mati == 1] = phi[1]
    phi_i_lijk[indexes.mati == 2] = phi[2]
    phi_i_lijk[indexes.mati == 3] = phi[3]

    phi_j_lijk[indexes.matj == 0] = phi[0]
    phi_j_lijk[indexes.matj == 1] = phi[1]
    phi_j_lijk[indexes.matj == 2] = phi[2]
    phi_j_lijk[indexes.matj == 3] = phi[3]

    return phi_i_lijk, phi_j_lijk, phi_i_lik, phi


def getddphi(parameters:object, indexes:object, quadrature_points):
    ddphi = np.array([-6 + 12 * quadrature_points,
                    -4 + 6 * quadrature_points,
                    6 - 12 * quadrature_points,
                    -2 + 6 * quadrature_points])

    ddphi_i_lijk = np.zeros((parameters.n, 4, 4, parameters.ns + 1))
    ddphi_j_lijk = np.zeros((parameters.n, 4, 4, parameters.ns + 1))

    ddphi_i_lijk[indexes.mati == 0] = ddphi[0]
    ddphi_i_lijk[indexes.mati == 1] = ddphi[1]
    ddphi_i_lijk[indexes.mati == 2] = ddphi[2]
    ddphi_i_lijk[indexes.mati == 3] = ddphi[3]

    ddphi_j_lijk[indexes.matj == 0] = ddphi[0]
    ddphi_j_lijk[indexes.matj == 1] = ddphi[1]
    ddphi_j_lijk[indexes.matj == 2] = ddphi[2]
    ddphi_j_lijk[indexes.matj == 3] = ddphi[3]

    return ddphi_i_lijk, ddphi_j_lijk


def geth(parameters):
    h_1D = np.ones(parameters.n) * parameters.l / parameters.n
    h_2D = h_1D[:, np.newaxis]
    h_3D = h_2D[:, np.newaxis]

    return h_1D, h_2D, h_3D


def getTinv(parameters, quadrature_points):
    x_l = np.arange(0, parameters.l, parameters.l / parameters.n)
    print("Aragne Problem getTinv")
    print(parameters.l)
    print(x_l)
    tinv_2D = np.outer(geth(parameters)[0], quadrature_points) + x_l[:, np.newaxis]
    print(f'n: {parameters.n}')
    print(f'n_tilde: {parameters.ns}')
    print(tinv_2D.shape)
    tinv_3D = tinv_2D[:, np.newaxis, :]
    tinv_4D = tinv_2D[:, np.newaxis, np.newaxis, :]

    return tinv_3D, tinv_4D


'''
Aufgabe 19 
'''


def getMbar_Aufgabe19(parameters, indexes, quadrature_points):
    h_3d = geth(parameters)[2]  # 0 = 1D, 1 = 2D, 2 = 3D
    exp = getexp(parameters)[0] + 1
    factors = np.power(h_3d, exp)

    phi_arrays = getphi(parameters, indexes, quadrature_points)
    vectorized_my = np.vectorize(parameters.my)
    integrand = vectorized_my(getTinv(parameters, quadrature_points)[1]) * phi_arrays[0] * phi_arrays[1]
    integral = np.dot(integrand, getstencil(quadrature_points))

    m_element = integral * factors

    return m_element


def getSbar_Aufgabe19(parameters, indexes, quadrature_points):
    h_3d = geth(parameters)[2]  # 0 = 1D, 1 = 2D, 2 = 3D
    exp = getexp(parameters)[0] - 3
    factors = np.power(h_3d, exp)

    ddphi_arrays = getddphi(parameters, indexes, quadrature_points)
    vectorized_E = np.vectorize(parameters.E)
    vectorized_I = np.vectorize(parameters.E)
    t_inv = getTinv(parameters, quadrature_points)[1]   # 4D array

    integrand = vectorized_E(t_inv)*vectorized_I(t_inv)*ddphi_arrays[0]*ddphi_arrays[1]
    integral = np.dot(integrand, getstencil(quadrature_points))    # axis 3 means sum along k dimension

    s_element = integral * factors

    return s_element


def getqbar_Aufgabe19(parameters, indexes, quadrature_points):
    h_2D = geth(parameters)[1]
    exp = getexp(parameters)[1] + 1     # 0 = 3D, 1 = 2D
    factors = np.power(h_2D, exp)

    phi_arrays = getphi(parameters, indexes, quadrature_points)
    vectorized_q = np.vectorize(parameters.q)
    t_inv = getTinv(parameters, quadrature_points)[0]

    integrand = vectorized_q(t_inv) * phi_arrays[2]
    integral = np.dot(integrand, getstencil(quadrature_points))

    q_element = factors * integral

    return q_element





if __name__ == "__main__":

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
            self.h = l/n
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
    alpha_e_static_solution_n3_arr = scipy.sparse.linalg.spsolve(getSe(params, idx_10), getve(params, idx_10))  # Find the static solution
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
    ax[0].plot(np.arange(0, params.l, params.l / (3 + 1)), alpha_e_static_solution_n3_arr[:2 * 3 + 2:2])
    ax[0].set_xlabel("x in m")
    ax[0].set_ylabel("w in m")
    ax[0].set_title(f"Solution for n = 3")
    ax[1].plot(np.arange(0, params.l, params.l / (params.n + 1)), alpha_e_static_solution_n100[:2 * params.n + 2:2])
    ax[1].set_title("Solution for n = 100")
    ax[1].set_ylabel("w in m")
    ax[1].set_xlabel("x in m")

    # plt.show()

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

    # getplot()

    '''
    Aufgabe 13
    '''

    params.set_q(x_0)
    n_error_test = 1000
    error_rates_plot = np.zeros(n_error_test)

    for n_iteration in range(1, n_error_test):
        params.set_n(n_iteration)
        idx_13 = Indices(getindizes(params))

        # Analytic Solution
        w_static_solution = np.zeros(2 * n_iteration + 2)  # Assemble Analytical solution like the numerical solution
        w_static_solution[::2] = analytical_w(params, np.linspace(0, l, n_iteration + 1))
        w_static_solution[1::2] = analytical_w_prime(params, np.linspace(0, l, n_iteration + 1))

        # Numeric Solution
        alpha_static_solution = scipy.sparse.linalg.spsolve(getSe(params, idx_13),
                                                            getve(params, idx_13))[:2 * n_iteration + 2]

        # Generate A matrix from the M matrix
        A = getM(params, idx_13)  # A == Mass-matrix for h = 1 and my = 1

        # Calculate the error
        numerator_error = ((w_static_solution - alpha_static_solution).T @ A) @ (
                    w_static_solution - alpha_static_solution)
        denominator_error = (w_static_solution.T @ A) @ w_static_solution
        relative_error = np.sqrt(numerator_error) / np.sqrt(denominator_error)

        error_rates_plot[n_iteration] = relative_error

        # Plot the Solutions (not part of Solution)
        '''
        fig, ax = plt.subplots(1, 2)

        ax[0].plot(np.linspace(0, 1, n_iteration + 1), alpha_static_solution[::2], c = "blue")
        ax[0].plot(np.arange(0, 1, 0.01), analytical_w(np.arange(0, 1, 0.01)), c = "orange")
        ax[1].plot(np.linspace(0, 1, n_iteration + 1), alpha_static_solution[1::2], c = "blue")
        ax[1].plot(np.arange(0, 1, 0.01), analytical_w_prime(np.arange(0, 1, 0.01)), c = "orange")
        plt.show()
        '''

    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Relative Error Rates over Varying number of Knots n")
    ax[0].plot(np.arange(n_error_test), error_rates_plot)
    ax[0].set_xlabel(f"n in 1")
    ax[0].set_ylabel("error_L^2 in 1")
    ax[1].plot(np.log(np.arange(n_error_test)), np.log(error_rates_plot))
    ax[1].set_xlabel(f'log(n) in 1')
    ax[1].set_ylabel("log(error_L^2) in 1")

    # plt.show()

    '''
    Aufgabe 14
    '''

    params.set_n(3)

    _, total_energy_a = newmark_simmulation(params, alpha_e_static_solution_n3_arr)
    params.eta = 1
    _, total_energy_b = newmark_simmulation(params, alpha_e_static_solution_n3_arr)
    params.beta = 1
    params.gamma = 1
    params.eta = 0.1
    _, total_energy_c = newmark_simmulation(params, alpha_e_static_solution_n3_arr)
    params.eta = 1
    _, total_energy_d = newmark_simmulation(params, alpha_e_static_solution_n3_arr)

    fig = plt.figure(figsize=(12, 6))

    sub_1 = plt.subplot(2, 2, 1)
    sub_1.set_ylabel("E_ges in Joule (J)")
    sub_1.set_xlabel("t in s")
    sub_1.set_title("Verlauf der Gesamtenergie für zeitliche Schrittweite eta = 0.1s \n beta = 0.25 und gamma = 0.5")
    sub_1.plot(np.arange(0, 0.1 * n_p, 0.1), total_energy_a)

    sub_2 = plt.subplot(2, 2, 2)
    sub_2.set_ylabel("E_ges in Joule (J)")
    sub_2.set_xlabel("t in s")
    sub_2.set_title("Verlauf der Gesamtenergie für zeitliche Schrittweite eta = 1s \n beta = 0.25 und gamma = 0.5")
    sub_2.plot(np.arange(0, 1 * n_p, 1), total_energy_b)

    sub_3 = plt.subplot(2, 2, 3)
    sub_3.set_ylabel("E_ges in Joule (J)")
    sub_3.set_xlabel("t in s")
    sub_3.set_title("Verlauf der Gesamtenergie für zeitliche Schrittweite eta = 0.1s \n beta = 1 und gamma = 1")
    sub_3.plot(np.arange(0, 0.1 * n_p, 0.1), total_energy_c)

    sub_4 = plt.subplot(2, 2, 4)
    sub_4.set_ylabel("E_ges in Joule (J)")
    sub_4.set_xlabel("t in s")
    sub_4.set_title("Verlauf der Gesamtenergie für zeitliche Schrittweite eta = 1s \n beta = 1 und gamma = 1")
    sub_4.plot(np.arange(0, 1 * n_p, 1), total_energy_d)

    sub_1.set_ylim(0, max(total_energy_a) * 1.2)
    sub_2.set_ylim(0, max(total_energy_a) * 1.2)
    sub_3.set_ylim(0, max(total_energy_a) * 1.2)
    sub_4.set_ylim(0, max(total_energy_a) * 1.2)

    plt.tight_layout()
    # plt.show()


    '''
    Aufgabe 16
    '''

    # The refrence points go from K_tilde_under = {0, ... n_tilde_under} (Formula 260) that's why we have the + 1
    reference_coordinates = np.linspace(0, 1, ns + 1)  # equidistant quadrature points

    print(getstencil(reference_coordinates))

    '''
    Aufgabe 18
    '''

    params.set_n(3)
    idx_18 = Indices(getindizes(params))
    print(getphi(params, idx_18, reference_coordinates)[1])


    '''
    Aufgabe 19
    '''

    idx_19 = Indices(getindizes(params))

    print(getMbar_Aufgabe19(params, idx_19, reference_coordinates))
    print(getSbar_Aufgabe19(params, idx_19, reference_coordinates))
    print(getqbar_Aufgabe19(params, idx_19, reference_coordinates))


    '''
    Aufgabe 20
    '''

    

    def getplot_Aufgabe20(parameters, indexes, solution_vector):
        h_2D = geth(parameters)[1]
        exp = getexp(parameters)[1]  # 0 = 3D, 1 = 2D
        factors = np.power(h_2D, exp)[0]    # because h_l is constant along the beam, i only need one factor

        # Maybe i need to update this to work with unevenly spaced h?

        k = np.arange(0, parameters.nh + 1)
        x_k = k / parameters.nh     # Spots were we approximate the function additionally (on refrence element)

        A_elements = getphi(parameters, indexes, x_k)[3].T * factors    # When h is constant the A matrix is constant for all l
        A_elements = np.tile(A_elements, (parameters.n, 1, 1))

        J, K = np.meshgrid(np.arange(4), k)     # Generate the indices

        J_indices = np.tile(J, (parameters.n, 1, 1))
        K_indices = np.tile(K, (parameters.n, 1, 1))
        L = np.arange(parameters.n).reshape(parameters.n, 1, 1) * np.ones((1, parameters.nh+1, 4)).astype(int)

        K_indices = (parameters.nh * L + K_indices).astype(int)     # Assemble after the given Formula

        J_indices = (2 * L + J_indices).astype(int)

        # If data gets assigns twice the coo_matrix sums the entries, that's why you have to filter duplicates
        mask = np.zeros_like(A_elements.flatten(), dtype=bool)
        mask[np.unique(K_indices.flatten() * A_elements.shape[1] + J_indices.flatten(), return_index=True)[1]] = True

        A = coo_matrix((A_elements.flatten()[mask], (K_indices.flatten()[mask], J_indices.flatten()[mask]))).tocsr()


        print(scipy.sparse.csr_matrix(solution_vector[:2*parameters.n+2]).shape)
        print(A.shape)

        print(A.toarray() @ solution_vector[:2*parameters.n+2])


        w = A.dot(scipy.sparse.csr_matrix(solution_vector[:2*parameters.n+2]))

        print(w.toarray())
        return



    # Example usage
    params.set_n(3)
    params.nh = 1
    idx_19 = Indices(getindizes(params))
    getplot_Aufgabe20(params, idx_19, alpha_e_static_solution_n3_arr)



