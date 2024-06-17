from math import sqrt, pi, atan
def Normalize(u):
    r = sqrt(u[0]**2 + u[1]**2 + u[2]**2)
    if not r: return u
    return [u[0]/r, u[1]/r, u[2]/r]

def arctan(fo9, t7t, lecos):
    if lecos>0 : return atan(fo9/t7t)
    return pi + atan(fo9/t7t)

def scalaire(v, u): return max(0,u[0]*v[0] + u[1]*v[1] + u[2]*v[2])

def diffVect(v, u):
    return [u[0]-v[0], u[1]-v[1], u[2]-v[2]]

def addVect(u, v):
    return [u[0]+v[0], u[1]+v[1], u[2]+v[2]]

def multiVect(u, x):
    return [u[0]*x, u[1]*x, u[2]*x]

def croi(X, Y):
    return [X[i]*Y[i] for i in range(len(X))]

def linearInterpolation( a, b, t):
    return addVect(multiVect(a,(1-t)) ,multiVect(b,t))

def norm(u):
    return sqrt(u[0]**2 + u[1]**2 + u[2]**2)

def sym(X, N):
    scal = scalaire(X, N)
    return [(X[i] - 2*scal*N[i]) for i in range(len(X)) ]

#Rendering Functions :
def D(roughness, N, H): #scalaire
    return (roughness**2)/(pi*((scalaire(N, H)**2)*(roughness**2 - 1) +1)**2 + 0.00001)

def G1(N, X, roughness): #scalaire
    k = (roughness**2)/2
    return scalaire(N,X)/(scalaire(N, X)*(1-k) + k)

def G(N, V, L, roughness): #scalaire
    return G1(N, V, roughness)*G1(N, L, roughness)

def F(F0, N, H): #Vecteur
    return F0 + multiVect(diffVect( F0, [1, 1, 1]), (1-scalaire(N, H))**5)

def Flamb(c): #vecteur
    return multiVect(c, 1/pi)

def Fspe( N, H, V, L, roughness): #scalaire
    return D(roughness, N, H)*G(N, V, L, roughness)/(4*scalaire(V, N)*scalaire(N, L) + 0.00001)

def BRDF(F0, N, H, V, L, roughness, c):
    Ks = F(F0, N, H)
    Kd = diffVect(Ks, [1, 1, 1])
    fd = croi(Kd, Flamb(c))
    fs = multiVect(Ks, Fspe(N, H, V, L, roughness))
    return addVect(fd, fs)