# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 12:16:04 2024

@author: carol
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
factorial = np.vectorize(np.math.factorial)

'''
Aufgabe 1, Initialisierung
'''

n = 3 # Anzahl der Elemente in 1
nh = 1 # Anzahl der zusätzlichen Auswertungspunkte je Element in 1
ns = 7 # Ordnung der Quadratur in 1
n_p = 100 # Anzahl der Zeitschritte in 1
beta = 1/4 # Newmark- Koeffizient in 1
gamma = 1/2 # Newmark- Koeffizient in 1
eta = 0.1 # Zeitschrittweite in s
l = 1 # Länge des Balkens in m
my = 1 # Längenspezifische Masse in kg/m
E = 1 # Elastizitätsmodul in N/m^2
I = 1 # Flächenträgheitsmoment in m^4
q = 1 # Streckenlast in N/m

B1 = [0, 1, 0] # Auslenkung linkes Ende in m
B2 = [0, 2, 0] # Anstieg linkes Ende in 1
B3 = [n, 3, 0] # Moment rechtes Ende in Nm
B4 = [n, 4, 0] # Querkraft rechtes Ende in N

B = np.array([B1, B2, B3, B4])


'''
Aufgabe 2, nützliche Arrays
'''
# a) 3D-Arrays
def getindizes():
    # a) 3D-Arrays
    nv = np.arange(0,4, 1) # erzeuge hilfsvektor [0, 1, 2, 3]
    J, I  = np.meshgrid(nv, nv) # erzeuge Wiederholungselemente 
    
    mati = np.array([I, I, I])# i Array
    matj = np.array([J, J, J])# j Array
    
    matl = np.repeat(np.arange(n).reshape(n, 1, 1), 4, axis = 1)
    matl = np.repeat(matl, 4, axis = 2) # l Array

    matlli = 2 * matl + mati # 2* l + i Array
    matllj = 2 * matl + matj # 2* l + j Array
    
    print("3D-Array [l]", matl)
    print("3D-Array [i]", mati)
    print("3D-Array [j]", matj)
    print("3D-Array [2l + i]", matlli)
    print("3D-Array [2l + j]", matllj)
    
    # b) 2D-Arrays
    vekl = np.repeat(np.arange(n).reshape(n, 1, 1), 4, axis = 1) # l Array
    i, j = np.meshgrid(nv, 1) # erzeuge Wiederholungselemente
    i = np.array(i).T # transponiere i
    veki = np.array([i, i, i]) # i Array
    veklli = 2* vekl + veki # 2* l + i Array
    
    print("2D-Array [l]", vekl)
    print("2D-Array [i]", veki)
    print("2D-Array [veklli]", veklli)
    
    return matl, mati, matj, matlli, matllj, vekl, veki, veklli
    

'''
Aufgabe 3,4; Elementmatrizen, -vektoren
'''   
def getMbar(h):
    faktor = 1
    matrix = np.array([[156, 22*h, 54, -13 * h], [22*h, 4* h**2, 13 * h, -3 * h**2], [54, 13*h, 156, -22*h], [-13*h, -3*h**2, -22*h, 4*h**2]])
    M = faktor * matrix
    M = np.tile(M, (n, 1, 1))
    return M

def getSbar(h):
    faktor = E*I / h**3
    matrix = np.array([[12, 6*h, -12, 6*h], [6*h, 4*h**2, -6*h, 2*h**2], [-12, -6*h, 12, -6*h], [6*h, 2*h**2, -6*h, 4*h**2]])
    S = faktor * matrix 
    S = np.tile(S, (n, 1, 1))
    return S

def getqbar(h):
    faktor = q*h /12
    vektor = np.array([[6], [h], [6], [-h]])
    vekq = faktor * vektor 
    vekq = np.tile(vekq, (n, 1, 1))
    return vekq

    

'''
Aufgabe 5, Massen-, Steifigkeitsmatrix, Streckenlastvektor
''' 
# indizes definieren
matl, mati, matj, matlli, matllj, vekl, veki, veklli = getindizes()

#Massenmatrix
def getM(h):
    M_alt = getMbar(h)#daten Matrix definieren
    M_neu = coo_matrix((M_alt.flatten(), (matlli.flatten(), matllj.flatten()))).tocsr() 
    #matlli.flatten() ist der 1 D array der Indizes der Zeilen wo Daten ungleich 0 sind
    #matllj.flatten() ist der 1 D array der Indizes der Spalten wo Daten ungleich 0 sind
    return M_neu

#Steifigkeitsmatrix
# analog zu getM für die Daten der Steifigkeitsmatrix 
def getS(h):
    S_alt = getSbar(h)
    S_neu = coo_matrix((S_alt.flatten(), (matlli.flatten(), matllj.flatten()))).tocsr()
    return S_neu
    

#Streckenlastvektor
# analog zu getM für werte des Streckenlastvektors
def getvq(h):
    vq_alt = getqbar(h)
    vq_neu = coo_matrix((vq_alt.flatten(), (veklli.flatten(), np.zeros_like(veklli.flatten())))).tocsr()
    #np.zeros_like(veklli.flatten()) erstellet ein Array aus Nullen, mit der gleichen Form wie veklli.flatten()
    return vq_neu

print(getM(1))
    
    
    




































