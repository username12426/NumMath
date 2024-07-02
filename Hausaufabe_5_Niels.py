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

x_0 = 1
x = 1   # Zufälliges x für die lamda Funktionen für Aufgabe 1 - 15

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


# We need this function, because when we change the n, the B matrix is changes as well
def generate_B(n_element):
    return np.array([[0, 1, 0],
                  [0, 2, 0],
                  [n_element, 3, 0],
                  [n_element, 4, 0]])


'''
Aufgabe 2, nützliche Arrays
'''

# a) 3D-Arrays
def getindizes(n_elements: int):
    # a) 3D-Arrays
    nv = np.arange(0, 4, 1)  # Create an array with values [0, 1, 2, 3]
    J, I = np.meshgrid(nv, nv)  #repeats the rows, columns of nv in J, I

    matl = np.arange(n_elements).reshape(n_elements, 1, 1) * np.ones((1, 4, 4)).astype(int) #Create a 3D array, where each element from 0 to n_elements-1 is repeated in shape(4,4)
    mati = np.repeat(I[np.newaxis, :, :], n_elements, axis=0) #Repeat the  array I along the first axis n_elements times
    matj = np.repeat(J[np.newaxis, :, :], n_elements, axis=0) #Repeat the  array J along the first axis n_elements times

    matlli = (2 * matl + mati).astype(int) #Calculate a 3D array from  matl & mati
    matllj = (2 * matl + matj).astype(int) #Calculate a 3D array from  matl & matj

    # We decided to use a row vector to represent the vector here
    # You can use a column vector as well [[[0], [0], [0], [0]], [[1], [1], ..
    # It does not matter that much as long as we stay consistent with this vector (matrix)
    # We use sparce matrices, so we flatten the vector anyway!
    veki, vekl = np.meshgrid(nv, np.arange(0, n_elements)) #Create 2D  arrays  using nv and an array from 0 to n_elements-1
    veklli = (2 * vekl + veki).astype(int) #Calculate a 2D array from  vekl + veki

    # Return the generated arrays
    return matl, mati, matj, matlli, matllj, vekl, veki, veklli


'''
Aufgabe 3,4; Elementmatrizen, -vektoren
'''

#create Mass matrix
def getMbar(h, my,  n_elements):
    faktor = my * h / 420 # define factor
    matrix = np.array(
        [[156, 22 * h, 54, -13 * h], [22 * h, 4 * h ** 2, 13 * h, -3 * h ** 2], [54, 13 * h, 156, -22 * h],
         [-13 * h, -3 * h ** 2, -22 * h, 4 * h ** 2]]) # define matrix
    M = faktor * matrix # scale matrix by the factor
    M = np.tile(M, (n_elements, 1, 1)) # Replicate the scaled matrix for each element
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


# Mass matrix
def getM(h, my, n_elements):
    M_alt = getMbar(h, my,  n_elements)  # Define the data matrix
    M_neu = coo_matrix((M_alt.flatten(), (matlli.flatten(), matllj.flatten()))).tocsr()
    return M_neu


# Stiffness matrix
# Analogous to getM for the data of the stiffness matrix
def getS(h, E, I, n_elements):
    S_alt = getSbar(h, E, I, n_elements)
    S_neu = coo_matrix((S_alt.flatten(), (matlli.flatten(), matllj.flatten()))).tocsr()# Convert the data matrix to a sparse matrix in COO format and then to CSR format
    return S_neu


# element vector
# Analogous to getM for the values of the element vector
def getvq(h, q, n_elements):
    vq_alt = getqbar(h, q, n_elements)
    vq_neu = coo_matrix((vq_alt.flatten(), (veklli.flatten(), np.zeros_like(veklli.flatten())))).tocsr()
    # np.zeros_like(veklli.flatten()) creates an array of zeros with the same shape as veklli.flatten()
    return vq_neu


'''
Aufgabe 6 Variante 1
'''

# Sind und nicht ganz sicher ob wir "==" verwenden dürfen, deshalb gubt es zwei Varianten
# B is not getting passed into the function, so we have to be careful when we redefine the B matrix!!

def getC(n_elements, B):

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

def getvn(n_elements, B):

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

def getMe(h, my, n_elements, B):
    M = getM(h, my, n_elements)
    C = getC(n_elements, B)
    C0 = np.zeros_like(C.toarray())
    I, J = np.meshgrid(np.arange(2), np.arange(2))
    zero_filler = coo_matrix((np.zeros(4), (I.flatten(), J.flatten()))).tocsr()

    M_C0_horizontal_stack = scipy.sparse.hstack([M, C0])
    filler_C0_horizontal_stack = scipy.sparse.hstack([C0.T, zero_filler])

    Me = scipy.sparse.vstack([M_C0_horizontal_stack, filler_C0_horizontal_stack])
    return Me


def getSe(h, E, I, n_elements, B):
    S = getS(h, E, I, n_elements)
    C = getC(n_elements, B)
    I, J = np.meshgrid(np.arange(2), np.arange(2))
    zero_filler = coo_matrix((np.zeros(4), (I.flatten(), J.flatten()))).tocsr()

    S_C_horizontal_stack = scipy.sparse.hstack([S, C])
    C_filler_horizontal_stack = scipy.sparse.hstack([C.T, zero_filler])
    Se = scipy.sparse.vstack([S_C_horizontal_stack, C_filler_horizontal_stack]).tocsr()
    return Se


def getve(h, q, n_elements, B):
    vq_nq = getvq(h, q, n_elements) + getvn(n_elements, B)
    v_E = scipy.sparse.vstack([vq_nq, getvd()])
    return v_E

'''
Aufgabe 10
'''

n = 3
h = l / n

B = generate_B(n)   # We changed the n to we have to update the B matrix
matl, mati, matj, matlli, matllj, vekl, veki, veklli = getindizes(n)    # Build the matricies for the DGL n = 3
alpha_e_static_solution_n3 = scipy.sparse.linalg.spsolve(getSe(h, E(x), I(x), n, B), getve(h, E(x), n, B))  # Find the static solution

'''
Aufgabe 11
'''

# Build the maticies for the DGL n = 100
n_100 = 100
h = l / n_100
B = generate_B(n_100)   # update the B matrix for the new n
matl, mati, matj, matlli, matllj, vekl, veki, veklli = getindizes(n_100)    # update index matrices
alpha_e_static_solution_n100 = scipy.sparse.linalg.spsolve(getSe(h, E(x), I(x), n_100, B), getve(h, q(x), n_100, B))

# Plot both solutions
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 4), sharey='row')
ax[0].plot(np.arange(0, l, l/(n+1)), alpha_e_static_solution_n3[:2*n+2:2])
ax[0].set_xlabel("x in m")
ax[0].set_ylabel("w in m")
ax[0].set_title(f"Solution for n = {n}")
ax[1].plot(np.arange(0, l, l/(n_100 + 1)), alpha_e_static_solution_n100[:2*n_100+2:2])
ax[1].set_title("Solution for n = 100")
ax[1].set_ylabel("w in m")
ax[1].set_xlabel("x in m")

plt.cla()
# plt.show()


'''
Aufgabe 12
'''


n = 3
q_static = 0   # Keine Streckenlast
h = l/n
B = generate_B(n)   # update the B matrix for the new n
matl, mati, matj, matlli, matllj, vekl, veki, veklli = getindizes(n)

M_e = getMe(h, my(x), n, B)
M = getM(h, my(x), n)
S = getS(h, E(x), I(x), n)
S_e = getSe(h, E(x), I(x), n, B)
v_e = getve(h, q_static, n, B)   # reinitialize the v_e vector, because the load has changed
v_q = getvq(h, q_static, n)
v_n = getvn(n, B)
C = getC(n, B)

def newmark_simmulation(timesteps, eta, beta, gamma):

    # use the static state as our startingpoint for the Newmark-algorithim

    a_0_rows = np.arange(len(alpha_e_static_solution_n3), dtype=int)
    a_0_cols = np.zeros_like(alpha_e_static_solution_n3, dtype=int)

    a_p = coo_matrix((alpha_e_static_solution_n3, (a_0_rows, a_0_cols))).tocsr()    # deflections

    # In the static case there is no acceleration and initial velocity
    a_d_p = coo_matrix((np.zeros_like(alpha_e_static_solution_n3), (a_0_rows, a_0_cols))).tocsr()   # velocities = 0
    a_dd_p = coo_matrix((np.zeros_like(alpha_e_static_solution_n3), (a_0_rows, a_0_cols))).tocsr()  # acceleration = 0


    a_p_animation = np.zeros((timesteps, 2*n+2))  # Data Matrix for the Animation
    total_energy_timesteps = np.zeros(timesteps)  # for Task 14

    # Iterate over n_p timesteps using the Newmark-Algorithm
    for time_step in range(timesteps):
        a_explicit = a_p + (a_d_p*eta) + (0.5 - beta)*(a_dd_p*eta**2)
        a_d_explicit = a_d_p + (1 - gamma)*(a_dd_p*eta)

        a_dd_p = scipy.sparse.linalg.spsolve(M_e + (S_e*beta*eta**2), v_e - S_e.dot(a_explicit))
        a_dd_p = coo_matrix((a_dd_p, (a_0_rows, a_0_cols))).tocsr()

        a_p = a_explicit + beta*a_dd_p*eta**2

        a_p_animation[time_step,:] = a_p.toarray()[:2*n+2].T    # only the first 2n+2 elements are coordinates

        a_d_p = a_d_explicit + gamma*a_dd_p*eta

        loads_v = a_p[2*n+2:]

        total_energy = 0.5*a_d_p[:2*n+2].T @ M @ a_d_p[:2*n+2] + (0.5*S*a_p[:2*n+2] - v_q - C @ loads_v - v_n).T @ a_p[:2*n+2]
        total_energy_timesteps[time_step] = total_energy[0].toarray()

    return a_p_animation, total_energy_timesteps

a_p_animation, total_energy_newmark = newmark_simmulation(n_p, eta, beta, gamma)



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


def analytical_w(x_k):
    return (q(x)/(E(x)*I(x))) * ((x_k**4)/24 - (l*x_k**3)/6 + ((l**2)*(x_k**2))/4)


def analytical_w_prime(x_k):
    return (q(x)/(E(x)*I(x))) * ((x_k**3)/6 - (l*x_k**2)/2 + (l**2*x_k)/2)


n_error_test = 1000
error_rates_plot = np.zeros(n_error_test)


for n_iteration in range(1, n_error_test):

    B = generate_B(n_iteration)
    h = l / n_iteration
    matl, mati, matj, matlli, matllj, vekl, veki, veklli = getindizes(n_iteration)

    # Analytic Solution
    w_static_solution = np.zeros(2*n_iteration+2)   # Assemble Analytical solution like the numerical solution
    w_static_solution[::2] = analytical_w(np.linspace(0, l, n_iteration+1))
    w_static_solution[1::2] = analytical_w_prime(np.linspace(0, l, n_iteration+1))

    # Numeric Solution
    alpha_static_solution = scipy.sparse.linalg.spsolve(getSe(h, E(x), I(x), n_iteration, B), getve(h, q(x), n_iteration, B))[:2*n_iteration+2]

    # Generate A matrix from the M matrix
    A = getM(1, 1, n_iteration)     # A == Mass-matrix for h = 1 and my = 1

    # Calculate the error
    numerator_error = ((w_static_solution - alpha_static_solution).T @ A) @ (w_static_solution - alpha_static_solution)
    denominator_error = (w_static_solution.T @ A) @ w_static_solution
    relative_error = np.sqrt(numerator_error)/np.sqrt(denominator_error)

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
plt.cla()

'''
Aufgabe 14
'''

_, total_energy_a = newmark_simmulation(n_p, eta, beta, gamma)
eta = 1
_, total_energy_b = newmark_simmulation(n_p, eta, beta, gamma)
beta = 1
gamma = 1
eta = 0.1
_, total_energy_c = newmark_simmulation(n_p, eta, beta, gamma)
eta = 1
_, total_energy_d = newmark_simmulation(n_p, eta, beta, gamma)


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
plt.cla()


'''
Aufgabe 15
'''

'''
Aufgabe 16
'''


# The refrence points go from K_tilde_under = {0, ... n_tilde_under} (Formula 260) that's why we have the + 1
reference_coordinates = np.linspace(0, 1, ns+1)   # equidistant quadrature points


def getstencil(quadrature_points):

    # Berechnung des stencils auf einem Referenzelement

    I, K = np.meshgrid(np.arange(len(quadrature_points)), quadrature_points)
    vandermonde_matrix = np.power(K, I)
    v_plus1 = 1 / (I[0] + 1)
    # We can interpret the formula for a as the solution of a system of equations
    s = np.linalg.solve(vandermonde_matrix.T, v_plus1)

    return s


print(getstencil(reference_coordinates))

'''
Aufgabe 17
'''


def getphi(quadrature_points):

    # calculate the base function values on a reference element
    # I don't understand why n is given in the task

    evaluated_points = np.array([[1 - 3*(quadrature_points**2) + 2*(quadrature_points**3)],
                                 [quadrature_points - 2*(quadrature_points**2) + quadrature_points**3],
                                 [3*(quadrature_points**2) - 2*(quadrature_points**3)],
                                 [-(quadrature_points**2) + (quadrature_points**3)]])

    return evaluated_points

print(getphi(reference_coordinates))



def getddphi(quadrature_points):
    evaluated_points = np.array([[-6 + 12*quadrature_points],
                                 [-4 + 6*quadrature_points],
                                 [6 - 12*quadrature_points],
                                 [-2 + 6*quadrature_points]])

    return evaluated_points

print(getddphi(reference_coordinates))


def geth(n_elements, beam_length):
    # This only works under the assumption that all beam elements are equally spaced
    # Wich they have to be because the mass matrix was derived using this assumption
    return np.ones(n_elements) * beam_length/n_elements


def getTinv(n_elements, beam_length, quadrature_points):
    # assuming an equidistant beam elements
    # x_l are the first n-1 knot positions (base points of the reference coordinates)
    x_l = np.arange(0, beam_length, beam_length/n_elements)
    return np.outer(geth(n_elements, beam_length), quadrature_points) + x_l[:, np.newaxis]

print(getTinv(3, 1, reference_coordinates))


def getexp(n_elements):
    i = np.arange(n_elements+1)
    j = np.arange(n_elements+1)
    J, I = np.meshgrid(j, i)
    delta_i_1 = (I == 1).astype(int)
    delta_i_3 = (I == 3).astype(int)
    delta_j_1 = (J == 1).astype(int)
    delta_j_3 = (J == 3).astype(int)

    exp_3d = np.stack([delta_i_1 + delta_i_3 + delta_j_1 + delta_j_3] * n_elements, axis=0)

    exp_2d = np.stack([delta_i_1[:, 0] + delta_i_3[:, 0]] * n_elements, axis=0)

    return exp_3d, exp_2d

print(getexp(3))













